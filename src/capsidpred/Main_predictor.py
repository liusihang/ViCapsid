# 文件名: model_pipeline.py
# 描述: 包含模型训练、评估和预测核心逻辑的模块。

import os
import sys
import json
import logging
import codecs
import shutil
import copy
import argparse

import torch
import torch.distributed
from subword_nmt.apply_bpe import BPE
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer

# --- 动态添加项目路径 ---
# 这使得从其他目录调用此脚本时，也能找到 'src' 目录
# 假设此文件位于项目根目录，或者调用脚本位于项目根目录
sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/SSFN"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/common"))

try:
    from common.metrics import metrics_multi_class, metrics_binary
    from common.multi_label_metrics import *
    from utils import set_seed, save_labels, get_parameter_number, load_trained_model
    from trainer import train
    from evaluater import evaluate
    from predictor import predict
    from data_loader import SequenceStructureProcessor
    from SSFN.model import *
except ImportError:
    # 兼容项目结构，假设 src 是顶级目录
    from src.common.metrics import metrics_multi_class, metrics_binary
    from src.common.multi_label_metrics import *
    from src.utils import set_seed, save_labels, get_parameter_number, load_trained_model
    from src.trainer import train
    from src.evaluater import evaluate
    from src.predictor import predict
    from src.data_loader import SequenceStructureProcessor
    from src.SSFN.model import SequenceAndStructureFusionNetwork

logger = logging.getLogger(__name__)


def _get_default_args():
    """
    内部辅助函数，定义并返回所有参数的默认值。
    """
    parser = argparse.ArgumentParser("Model Building for LucaProt")
    
    # --- Data and Task Arguments ---
    parser.add_argument("--data_dir", default=None, type=str, help="the dataset dirpath.")
    parser.add_argument("--separate_file", action="store_true", help="Load dataset with file names in CSV.")
    parser.add_argument("--filename_pattern", default=None, type=str, help="Dataset filename pattern.")
    parser.add_argument("--tfrecords", action="store_true", help="Whether the dataset is in tfrecords format.")
    parser.add_argument("--shuffle_queue_size", default=5000, type=int, help="Shuffle queue size for tfrecords.")
    parser.add_argument("--multi_tfrecords", action="store_true", help="Whether multiple tfrecords exist.")
    parser.add_argument("--dataset_name", default="rdrp_40_extend", type=str, help="Dataset name.")
    parser.add_argument("--dataset_type", default="protein", type=str, choices=["protein", "dna", "rna"], help="Dataset type.")
    parser.add_argument("--task_type", default="binary_class", type=str, choices=["multi_label", "multi_class", "binary_class"], help="Task type.")
    parser.add_argument("--model_type", default=None, type=str, choices=["sequence", "structure", "embedding", "sefn", "ssfn"], help="Model type.")
    parser.add_argument("--subword", action="store_true", help="Whether to use subword-level for sequence.")
    parser.add_argument("--codes_file", type=str, default="../subword/rdrp/protein_codes_rdrp_20000.txt", help="Subword codes filepath.")
    parser.add_argument("--input_mode", type=str, default="single", choices=["single", "concat", "independent"], help="The input operation.")
    parser.add_argument("--label_type", default="rdrp", type=str, help="Label type.")
    parser.add_argument("--label_filepath", default=None, type=str, help="The label list filepath.")
    
    # --- Structure Arguments ---
    parser.add_argument("--cmap_type", default=None, type=str, choices=["C_alpha", "C_bert"], help="Contact map calculation type.")
    parser.add_argument("--cmap_thresh", default=10.0, type=float, help="Contact map threshold.")
    
    # --- Directory Arguments ---
    parser.add_argument("--output_dir", default=None, type=str, help="The output dirpath.")
    parser.add_argument("--log_dir", default="./logs/", type=str, help="Log dir.")
    parser.add_argument("--tb_log_dir", default="./tb-logs/", type=str, help="Tensorboard log dir.")
    
    # --- Model Config Arguments ---
    parser.add_argument("--config_path", default=None, type=str, help="Config filepath of the running model.")
    parser.add_argument("--seq_vocab_path", default=None, type=str, help="Sequence token vocab filepath.")
    parser.add_argument("--struct_vocab_path", default=None, type=str, help="Structure node token vocab filepath.")
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache dirpath.")
    
    # --- Pooling and Activation Arguments ---
    parser.add_argument("--seq_pooling_type", type=str, default=None, choices=["none", "sum", "max", "avg", "attention", "context_attention", "weighted_attention", "value_attention", "transformer"], help="Pooling for sequence encoder.")
    parser.add_argument("--struct_pooling_type", type=str, default=None, choices=["sum", "max", "avg", "attention", "context_attention", "weighted_attention", "value_attention", "transformer"], help="Pooling for structure encoder.")
    parser.add_argument("--embedding_pooling_type", type=str, default=None, choices=["none", "sum", "max", "avg", "attention", "context_attention", "weighted_attention", "value_attention", "transformer"], help="Pooling for embedding encoder.")
    parser.add_argument("--activate_func", type=str, default=None, choices=["tanh", "relu", "leakyrelu", "gelu"], help="Activation function after pooling.")

    # --- Execution Control Arguments ---
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Evaluate during training.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set for uncased models.")

    # --- Training Hyperparameters ---
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Total number of training steps. Overrides num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints.")
    
    # --- Environment Arguments ---
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached sets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fp16", action="store_true", help="Use 16-bit (mixed) precision.")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="Apex AMP optimization level.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")

    # --- Loss and Metric Arguments ---
    parser.add_argument("--sigmoid", action="store_true", help="Add sigmoid in classifier.")
    parser.add_argument("--loss_type", type=str, default="bce", choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"], help="Loss type.")
    parser.add_argument("--max_metric_type", type=str, default="f1", choices=["acc", "jaccard", "prec", "recall", "f1", "fmax", "roc_auc", "pr_auc"], help="Metric for model selection.")
    parser.add_argument("--pos_weight", type=float, default=40, help="Positive weight for BCE.")
    parser.add_argument("--weight", type=str, default=None, help="Weights for multi-class CE.")
    parser.add_argument("--focal_loss_alpha", type=float, default=0.7, help="Focal loss alpha.")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument("--focal_loss_reduce", action="store_true", help="Mean for one sample in focal loss.")
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0, help="Negative gamma for ASL.")
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0, help="Positive gamma for ASL.")

    # --- Input Length and Embedding Arguments ---
    parser.add_argument("--seq_max_length", default=2048, type=int, help="Max sequence length.")
    parser.add_argument("--struct_max_length", default=2048, type=int, help="Max contact map length.")
    parser.add_argument("--trunc_type", default="right", type=str, choices=["left", "right"], help="Truncation type.")
    parser.add_argument("--no_position_embeddings", action="store_true", help="Do not use position embeddings.")
    parser.add_argument("--no_token_type_embeddings", action="store_true", help="Do not use token type embeddings.")
    parser.add_argument("--embedding_input_size", default=2560, type=int, help="Input embedding dimension.")
    parser.add_argument("--embedding_type", type=str, default="matrix", choices=[None, "contacts", "bos", "matrix"], help="Type of structural embedding info.")
    parser.add_argument("--embedding_max_length", default=2048, type=int, help="Max embedding length.")

    # --- Model Loading and Saving ---
    parser.add_argument("--model_dirpath", default=None, type=str, help="Load a trained model to continue training.")
    parser.add_argument("--save_all", action="store_true", help="Save all checkpoints.")
    parser.add_argument("--delete_old", action="store_true", help="Delete old checkpoints based on metric.")

    args = parser.parse_args([])
    return args


def run_model_pipeline(config_overrides: dict):
    """
    Executes the model training, evaluation, and prediction pipeline.

    Args:
        config_overrides (dict): A dictionary of parameters to override the defaults.
                                 Keys should match the argument names (e.g., 'data_dir').
    
    Returns:
        str or None: Path to the best model if training was performed, otherwise None.
    """
    # 1. Get default arguments and update with user-provided overrides
    args_namespace = _get_default_args()
    args_dict = vars(args_namespace)
    args_dict.update(config_overrides)
    # Check for required arguments
    required_args = ['data_dir', 'dataset_name', 'task_type', 'model_type', 'label_type',
                     'label_filepath', 'output_dir', 'log_dir', 'tb_log_dir', 'config_path',
                     'max_metric_type', 'trunc_type']
    for req in required_args:
        if args_dict.get(req) is None:
            raise ValueError(f"Missing required argument in configuration: '{req}'")

    args = argparse.Namespace(**args_dict)

    # 2. Start of the original main() logic
    if args.model_type == "sequence":
        output_input_col_names = [args.dataset_type, "seq"]
        args.has_seq_encoder = True
        args.has_struct_encoder = False
        args.has_embedding_encoder = False
        args.cmap_type = None
        args.embedding_type = None
    elif args.model_type == "structure":
        output_input_col_names = [args.dataset_type, "structure"]
        args.has_seq_encoder = False
        args.has_struct_encoder = True
        args.has_embedding_encoder = False
        args.embedding_type = None
    elif args.model_type == "embedding":
        output_input_col_names = [args.dataset_type, "embedding"]
        args.has_seq_encoder = False
        args.has_struct_encoder = False
        args.has_embedding_encoder = True
        args.cmap_type = None
    elif args.model_type == "sefn":
        output_input_col_names = [args.dataset_type, "seq", "embedding"]
        args.has_seq_encoder = True
        args.has_struct_encoder = False
        args.has_embedding_encoder = True
        args.cmap_type = None
    elif args.model_type == "ssfn":
        output_input_col_names = [args.dataset_type, "seq", "structure"]
        args.has_seq_encoder = True
        args.has_struct_encoder = True
        args.has_embedding_encoder = False
        args.embedding_type = None
    else:
        raise Exception("Not support this model_type=%s" % args.model_type)

    is_master = (args.local_rank in [-1, 0])

    if is_master:
        if (os.path.exists(args.output_dir)
                and os.listdir(args.output_dir)
                and args.do_train
                and not args.overwrite_output_dir):
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(args.output_dir, exist_ok=True)
    if args.local_rank != -1:
        torch.distributed.barrier()
        
    os.makedirs(args.log_dir, exist_ok=True)
    log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    os.makedirs(args.tb_log_dir, exist_ok=True)

    log_fp.write("Inputs:\n")
    log_fp.write("Input Name List: %s\n" % ",".join(output_input_col_names))
    log_fp.write("#" * 50 + "\n")

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.n_gpu = torch.cuda.device_count()
        if args.n_gpu > 1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            if args.local_rank == 0:
                print('world size: %d' % torch.distributed.get_world_size())
        else:
            device = torch.device("cuda")
    args.device = device

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s | %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        force=True # Override any existing logger configs
    )
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )

    set_seed(args)
    args.dataset_name = args.dataset_name.lower()
    processor = SequenceStructureProcessor(
        model_type=args.model_type,
        separate_file=args.separate_file,
        filename_pattern=args.filename_pattern
    )

    args.output_mode = args.task_type
    if args.output_mode in ["multi_label", "binary_class", "binary-class"]:
        args.sigmoid = True

    label_list = processor.get_labels(label_filepath=args.label_filepath)
    num_labels = len(label_list)
    logger.info("Num Labels: %d", num_labels)
    if is_master:
        save_labels(os.path.join(args.log_dir, "label.txt"), label_list)

    args.model_type = args.model_type.lower()
    config_class = BertConfig
    config = config_class(**json.load(open(args.config_path, "r")))
    config.max_position_embeddings = int(args.seq_max_length)
    config.num_labels = num_labels
    config.embedding_pooling_type = args.embedding_pooling_type
    if args.activate_func:
        config.activate_func = args.activate_func
    if args.pos_weight:
        config.pos_weight = args.pos_weight

    subword = None
    seq_tokenizer_class = None
    seq_tokenizer = None
    if args.has_seq_encoder:
        if not args.seq_vocab_path: raise ValueError("'seq_vocab_path' is required for models with a sequence encoder.")
        seq_tokenizer_class = BertTokenizer
        seq_tokenizer = seq_tokenizer_class(args.seq_vocab_path, do_lower_case=args.do_lower_case)
        config.vocab_size = seq_tokenizer.vocab_size
        if args.subword:
            with codecs.open(args.codes_file) as bpe_codes_prot:
                subword = BPE(bpe_codes_prot, merges=-1, separator='')

    struct_tokenizer_class = None
    struct_tokenizer = None
    if args.has_struct_encoder:
        if not args.struct_vocab_path: raise ValueError("'struct_vocab_path' is required for models with a structure encoder.")
        struct_tokenizer_class = BertTokenizer
        struct_tokenizer = struct_tokenizer_class(args.struct_vocab_path, do_lower_case=args.do_lower_case)
        config.struct_vocab_size = struct_tokenizer.vocab_size

    model_class = SequenceAndStructureFusionNetwork
    if args.model_dirpath and os.path.exists(args.model_dirpath):
        model = load_trained_model(config, args, model_class, args.model_dirpath)
    else:
        model = model_class(config, args)

    if is_master:
        str_config = copy.deepcopy(config)
        if hasattr(config, 'id2label') and len(config.id2label) > 10:
            str_config.id2label = "..."
            str_config.label2id = "..."
        log_fp.write("Model Config:\n %s\n" % str(str_config))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Model Architecture:\n %s\n" % str(model))
        log_fp.write("#" * 50 + "\n")

    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.local_rank == 0:
        torch.distributed.barrier()

    if is_master:
        logger.info("====Training/Evaluation Parameters:=====")
        args_dict_for_log = {k: v for k, v in vars(args).items() if k != "device"}
        for attr, value in sorted(args_dict_for_log.items()):
            logger.info("\t{}={}".format(attr, value))
        logger.info("====Parameters End=====\n")

        log_fp.write(json.dumps(args_dict_for_log, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        log_fp.write("num labels: %d\n" % num_labels)
        log_fp.write("#" * 50 + "\n")

        model_size_info = get_parameter_number(model)
        log_fp.write(json.dumps(model_size_info, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        log_fp.flush()

    max_metric_model_info = None
    if args.do_train:
        logger.info("++++++++++++Training+++++++++++++")
        global_step, tr_loss, max_metric_model_info = train(
            args, model, processor, seq_tokenizer, subword, struct_tokenizer=struct_tokenizer, log_fp=log_fp
        )
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and is_master:
        logger.info("++++++++++++Save Model+++++++++++++")
        best_output_dir = os.path.join(args.output_dir, "best")
        global_step = max_metric_model_info["global_step"]
        prefix = f"checkpoint-{global_step}"
        source_dir = os.path.join(args.output_dir, prefix)
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, best_output_dir, dirs_exist_ok=True)
            logger.info("Saving model checkpoint to %s", best_output_dir)
            torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
            save_labels(os.path.join(best_output_dir, "label.txt"), label_list)
        else:
            logger.error(f"Checkpoint directory not found: {source_dir}")


    if args.do_eval and is_master:
        logger.info("++++++++++++Validation+++++++++++++")
        log_fp.write("++++++++++++Validation+++++++++++++\n")
        if not max_metric_model_info: raise ValueError("Evaluation requires a trained model. 'max_metric_model_info' is missing. Please run with 'do_train=True'.")
        global_step = max_metric_model_info["global_step"]
        logger.info("max %s global step: %d", args.max_metric_type, global_step)
        log_fp.write(f"max {args.max_metric_type} global step: {global_step}\n")
        prefix = f"checkpoint-{global_step}"
        checkpoint = os.path.join(args.output_dir, "best") # Always evaluate the best one
        model = model_class.from_pretrained(checkpoint, args=args)
        model.to(args.device)
        result = evaluate(args, model, processor, seq_tokenizer, subword, struct_tokenizer, prefix=prefix, log_fp=log_fp)
        logger.info("Evaluation results: %s", json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    if args.do_predict and is_master:
        logger.info("++++++++++++Testing+++++++++++++")
        log_fp.write("++++++++++++Testing+++++++++++++\n")
        if not max_metric_model_info: raise ValueError("Prediction requires a trained model. 'max_metric_model_info' is missing. Please run with 'do_train=True'.")
        global_step = max_metric_model_info["global_step"]
        logger.info("max %s global step: %d", args.max_metric_type, global_step)
        log_fp.write(f"max {args.max_metric_type} global step: {global_step}\n")
        prefix = f"checkpoint-{global_step}"
        checkpoint = os.path.join(args.output_dir, "best") # Always predict with the best one
        model = model_class.from_pretrained(checkpoint, args=args)
        model.to(args.device)
        _, _, result = predict(args, model, processor, seq_tokenizer, subword, struct_tokenizer, prefix=prefix, log_fp=log_fp)
        logger.info("Prediction results: %s", json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    if is_master and log_fp:
        log_fp.close()

    if args.n_gpu > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    logger.info("Pipeline finished.")
    if args.do_train and is_master:
        return os.path.join(args.output_dir, "best")
    return None

if __name__ == '__main__':
    print("This is a library module. Please import 'run_model_pipeline' from it into your main script.")
    print("Example Usage: from model_pipeline import run_model_pipeline; run_model_pipeline(config_dict)")



'''
# 文件名: run_experiment.py
# 描述: 定义并运行一个或多个模型训练实验。

import os
import json
from model_pipeline import run_model_pipeline


def main():
    """主函数，用于定义和运行一个单一实验。"""

    # 1. 定义您的实验配置
    # 所有必需的参数都必须在这里提供。
    experiment_config = {
        # --- 必需的路径和类型信息 ---
        "data_dir": data_dir,
        "label_filepath": label_path,
        "config_path": config_path,
        "seq_vocab_path": vocab_path,
        "output_dir": "./experiment_results/sequence_model_run",
        "log_dir": "./experiment_results/sequence_model_run/logs",
        "tb_log_dir": "./experiment_results/sequence_model_run/tb_logs",
        
        # --- 任务和模型定义 ---
        "dataset_name": "demo_protein_classification",
        "dataset_type": "protein",
        "task_type": "binary_class",
        "model_type": "sequence", # 试试 'sequence' 或 'ssfn' (需要 struct_vocab_path)
        "label_type": "binary",
        "max_metric_type": "f1",
        "trunc_type": "right",
        
        # --- 训练控制 ---
        "do_train": True,
        "do_eval": True,
        "do_predict": True,
        "overwrite_output_dir": True,
        "evaluate_during_training": True,
        
        # --- 超参数调整 (为快速演示设置较小值) ---
        "num_train_epochs": 2,
        "learning_rate": 5e-5,
        "per_gpu_train_batch_size": 2,
        "per_gpu_eval_batch_size": 2,
        "logging_steps": 1,
        "save_steps": 1,
        "seed": 42,

        # --- 数据加载器设置 ---
        "filename_pattern": "{}.csv", # 告诉加载器文件名格式
        "separate_file": True, # 因为我们有 train/dev/test 子目录
    }

    # 如果模型是 ssfn，需要提供结构词汇表
    if experiment_config["model_type"] == "ssfn":
        experiment_config["struct_vocab_path"] = vocab_path

    print("\n--- Starting Experiment ---")
    print("Configuration:")
    print(json.dumps(experiment_config, indent=2))
    print("----------------------------")

    # 2. 调用流水线函数并传入配置
    try:
        best_model_path = run_model_pipeline(config_overrides=experiment_config)
        if best_model_path:
            print(f"\n✅ Experiment finished successfully!")
            print(f"   Best model saved to: {best_model_path}")
    except Exception as e:
        import traceback
        print(f"\n❌ An error occurred during the pipeline execution: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
'''