#!/usr/bin/env python
# encoding: utf-8
'''
@author: 您的名字
@desc: 一个简化的批量预测脚本，用于加载训练好的模型并处理一个文件夹中所有的TFRecord文件。
'''
import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader

from transformers.models.bert.tokenization_bert import BertTokenizer
from subword_nmt.apply_bpe import BPE
import codecs


from .data_loader import parse_tfrecord
from tfrecord.torch.dataset import TFRecordDataset
from .SSFN.model import SequenceAndStructureFusionNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """
    自定义的collate_fn，用于处理包含字符串ID的批次数据。
    """
    # batch 是一个列表, e.g., [ (features_1, id_1), (features_2, id_2), ... ]
    # features_x 是一个包含多个张量的元组, id_x 是字符串
    
    # 分离特征和ID
    feature_tuples = [item[:-1] for item in batch]
    ids = [item[-1] for item in batch] # 现在这里得到的是一个字符串列表

    # 使用默认的collate函数来堆叠所有特征张量
    collated_features = torch.utils.data.dataloader.default_collate(feature_tuples)
    
    # 返回批处理好的特征和ID列表
    return collated_features, ids


def predictor(args):
    """
    主批量预测函数
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"正在使用 {str(device).upper()} 进行预测。")
    args.device = device

    # 2. 加载训练参数和配置
    logger.info(f"从模型目录加载配置: {args.model_dir}")
    training_args_path = os.path.join(args.model_dir, "training_args.bin")
    if not os.path.exists(training_args_path):
        raise FileNotFoundError(f"在 {args.model_dir} 中未找到 training_args.bin 文件。")
    train_args = torch.load(training_args_path, weights_only=False) # 在较新版本PyTorch中建议设置weights_only
    train_args.device = device
    
    # 3. 加载模型、分词器和标签
    model_class = SequenceAndStructureFusionNetwork
    logger.info("正在加载已训练好的模型...")
    model = model_class.from_pretrained(args.model_dir, args=train_args)
    model.to(device)
    model.eval()

    seq_tokenizer, subword, struct_tokenizer = None, None, None
    if train_args.has_seq_encoder:
        seq_tokenizer_path = os.path.join(args.model_dir, "sequence")
        if not os.path.isdir(seq_tokenizer_path): seq_tokenizer_path = args.model_dir
        logger.info(f"正在从 '{seq_tokenizer_path}' 加载序列分词器...")
        seq_tokenizer = BertTokenizer.from_pretrained(seq_tokenizer_path, do_lower_case=train_args.do_lower_case)
        if train_args.subword:
            logger.info("正在加载 BPE 编码...")
            bpe_codes_prot = codecs.open(train_args.codes_file)
            subword = BPE(bpe_codes_prot, merges=-1, separator='')

    if train_args.has_struct_encoder:
        struct_tokenizer_path = os.path.join(args.model_dir, "structure")
        if not os.path.isdir(struct_tokenizer_path): struct_tokenizer_path = args.model_dir
        logger.info(f"正在从 '{struct_tokenizer_path}' 加载结构分词器...")
        struct_tokenizer = BertTokenizer.from_pretrained(struct_tokenizer_path, do_lower_case=train_args.do_lower_case)

    label_filepath = os.path.join(args.model_dir, "label.txt")
    if os.path.exists(label_filepath):
        with open(label_filepath, "r", encoding="utf-8") as f:
            label_list = [line.strip() for line in f if line.strip()]
        logger.info(f"已加载 {len(label_list)} 个标签。")
    else:
        label_list = list(model.config.id2label.values())
        logger.warning("未找到 label.txt。正在使用模型配置中的标签。")
    label_map = {label: i for i, label in enumerate(label_list)}

    # 4. 准备批量预测
    tfrecord_files = glob(os.path.join(args.input_dir, "*.tfrecords"))
    tfrecord_files += glob(os.path.join(args.input_dir, "*.records"))
    
    if not tfrecord_files:
        logger.error(f"在目录 '{args.input_dir}' 中没有找到任何 .tfrecord 或 .records 文件。")
        return
    logger.info(f"发现 {len(tfrecord_files)} 个待处理文件。")
    
    all_results = []

    # 5. 循环处理所有文件
    for i, file_path in enumerate(tfrecord_files):
        logger.info(f"--- 正在处理文件 {i+1}/{len(tfrecord_files)}: {os.path.basename(file_path)} ---")
        
        index_path = file_path.replace(".tfrecords", ".index").replace(".records", ".index")
        if not os.path.exists(index_path):
            logger.warning(f"警告：未找到索引文件 '{index_path}'。对于大数据集，这可能会导致性能下降。")
            index_path = None

        dataset = TFRecordDataset(
            file_path,
            index_path=index_path,
            description=None,
            shuffle_queue_size=None,
            transform=lambda x: parse_tfrecord(
                x, subword=subword, seq_tokenizer=seq_tokenizer, struct_tokenizer=struct_tokenizer,
                seq_max_length=train_args.seq_max_length, struct_max_length=train_args.struct_max_length,
                embedding_type=train_args.embedding_type, embedding_max_length=train_args.embedding_max_length,
                output_mode=train_args.output_mode, label_map=label_map, pad_on_left=False,
                pad_token=seq_tokenizer.convert_tokens_to_ids([seq_tokenizer.pad_token])[0] if seq_tokenizer else 0,
                pad_token_segment_id=0, mask_padding_with_zero=True, trunc_type=train_args.trunc_type,
                cmap_type=train_args.cmap_type, cmap_thresh=train_args.cmap_thresh
            )
        )
        
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn,num_workers=4,pin_memory=True,prefetch_factor=2,drop_last=False)

        with torch.no_grad():
            for feature_batch, id_batch in data_loader:
                input_dict = {}
                feature_idx = 0
                if train_args.has_seq_encoder:
                    input_dict['input_ids'] = feature_batch[feature_idx].to(device)
                    input_dict['attention_mask'] = feature_batch[feature_idx+1].to(device)
                    input_dict['token_type_ids'] = feature_batch[feature_idx+2].to(device)
                    feature_idx += 4
                if train_args.has_struct_encoder:
                    input_dict['struct_input_ids'] = feature_batch[feature_idx].to(device)
                    input_dict['struct_contact_map'] = feature_batch[feature_idx+1].to(device)
                    feature_idx += 3
                if train_args.embedding_type:
                    input_dict['embedding_info'] = feature_batch[feature_idx].to(device)
                    if train_args.embedding_type != "bos":
                        input_dict['embedding_attention_mask'] = feature_batch[feature_idx+1].to(device)
                        feature_idx += 2
                    else:
                        feature_idx += 1

                outputs = model(**input_dict)
                logits = outputs[0]

                if train_args.sigmoid: probs = torch.sigmoid(logits).cpu().numpy()
                else: probs = torch.softmax(logits, dim=-1).cpu().numpy()

                if train_args.task_type == 'binary_class': preds = (probs > 0.5).astype(int)
                elif train_args.task_type == 'multi_label': preds = (probs > 0.5).astype(int)
                else: preds = np.argmax(probs, axis=1)

                for j in range(len(id_batch)):
                    # 在每次循环开始时，包裹一个 try...except 块
                    try:
                        result = {'id': id_batch[j], 'source_file': os.path.basename(file_path)}
                        
                        if train_args.task_type == 'binary_class':
                            if len(label_list) == 2 and probs.shape[1] == 1:
                                negative_label, positive_label = label_list[0], label_list[1]
                                positive_prob = probs[j][0]
                                result[f'prob_{positive_label}'] = positive_prob
                                result[f'prob_{negative_label}'] = 1.0 - positive_prob
                                prediction_index = preds[j][0]
                                result['prediction_index'] = prediction_index
                                result['prediction_label'] = label_list[prediction_index]
                            else:
                                for k, label in enumerate(label_list):
                                    result[f'prob_{label}'] = probs[j][k]
                                    # 注意：在二分类中，preds的形状可能是 (batch_size,)
                                    # 所以这里也可能出错，我们先专注于 probs
                                    result[f'pred_{label}'] = preds[j][k] if preds.ndim > 1 else preds[j]

                        elif train_args.task_type == 'multi_label':
                            for k, label in enumerate(label_list):
                                result[f'prob_{label}'] = probs[j][k]
                                result[f'pred_{label}'] = preds[j][k]
                                
                        else: # multi_class
                            prediction_index = preds[j]
                            result['prediction_index'] = prediction_index
                            result['prediction_label'] = label_list[prediction_index]
                            for k, label in enumerate(label_list):
                                result[f'prob_{label}'] = probs[j][k]
                        
                        all_results.append(result)

                    except IndexError:
                        # ===================== 这是最重要的诊断部分 =====================
                        print("\n" + "="*80, file=sys.stderr)
                        print("!!! CAUGHT IndexError: 正在打印导致崩溃的变量状态 !!!", file=sys.stderr)
                        print(f"--- 任务信息 ---", file=sys.stderr)
                        print(f"train_args.task_type: {train_args.task_type}", file=sys.stderr)
                        print(f"label_list: {label_list} (长度: {len(label_list)})", file=sys.stderr)
                        
                        print(f"\n--- 批次数据状态 (在样本 #{j} 处崩溃) ---", file=sys.stderr)
                        print(f"ID: {id_batch[j]}", file=sys.stderr)
                        
                        print(f"\n--- 数组形状诊断 ---", file=sys.stderr)
                        print(f"整个 probs 数组的形状 (probs.shape): {probs.shape}", file=sys.stderr)
                        print(f"整个 preds 数组的形状 (preds.shape): {preds.shape}", file=sys.stderr)
                        
                        print(f"\n--- 导致错误的具体数据 ---", file=sys.stderr)
                        print(f"当前样本的概率 (probs[j]): {probs[j]}", file=sys.stderr)
                        print(f"当前样本的预测 (preds[j]): {preds[j]}", file=sys.stderr)
                        
                        print(f"\n--- 循环变量 ---", file=sys.stderr)
                        # 为了捕获 k 的值，我们需要把循环也包进来，或者在这里手动模拟
                        print("尝试访问的索引 k 导致了错误。由于错误发生在循环内部，我们知道它至少是1。", file=sys.stderr)
                        
                        print("\n" + "="*80, file=sys.stderr)
                        
                        # 在打印完所有信息后，重新抛出异常，让程序按预期停止
                        raise
                        # =================================================================
    
    # 6. 结果汇总与保存
    if not all_results:
        logger.warning("所有文件中都没有样本，或处理失败，未生成任何预测结果。")
        return
    logger.info(f"汇总所有结果并保存到 {args.output_file}...")
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    logger.info("任务完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用已训练的模型批量处理一个文件夹中的所有TFRecord文件。")
    parser.add_argument("--model_dir", type=str, required=True, help="已训练模型的目录路径 (例如 'best' 目录)。")
    parser.add_argument("--input_dir", type=str, required=True, help="包含待预测TFRecord文件的文件夹路径。")
    parser.add_argument("--output_file", type=str, required=True, help="保存所有预测结果的汇总文件路径 (CSV格式)。")
    parser.add_argument("--batch_size", type=int, default=32, help="预测时使用的批处理大小。")
    parser.add_argument("--no_cuda", action="store_true", help="禁用CUDA，在CPU上运行。")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parser.parse_args()
    predictor(args)