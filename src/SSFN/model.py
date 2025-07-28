#!/usr/bin/env python
# encoding: utf-8
'''
*Copyright (c) 2023, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.

@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/19 14:23
@project: DeepProtFunc
@file: model
@desc: our model(DeepProtFunc)
'''
import copy, sys
import logging
logger = logging.getLogger(__name__)
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from pooling import *
    from modeling_bert import BertModel, BertPreTrainedModel
    from common.loss import AsymmetricLossOptimized, FocalLoss, MultiLabel_CCE
    from gcn import *
except ImportError:
    from src.SSFN.pooling import *
    from src.SSFN.modeling_bert import BertModel, BertPreTrainedModel
    from src.common.loss import AsymmetricLossOptimized, FocalLoss, MultiLabel_CCE
    from src.SSFN.gcn import *
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss


def create_pooler(pooler_type, config, args):
    '''
    pooler building
    :param pooler_type:
    :param config:
    :param args:
    :return:
    '''
    if pooler_type == "seq":
        pooling_type = args.seq_pooling_type
        hidden_size = config.hidden_size
    elif pooler_type == "struct":
        pooling_type = args.struct_pooling_type
        hidden_size = sum(config.struct_output_size)
        if pooling_type is None:
            pooling_type = "sum"
    elif pooler_type == "embedding":
        pooling_type = args.embedding_pooling_type
        hidden_size = config.embedding_input_size
    else:
        raise Exception("Not support pooler_type=%s" % pooler_type)

    if pooling_type == "max":
        return GlobalMaskMaxPooling1D()
    elif pooling_type == "sum":
        return GlobalMaskSumPooling1D(axis=1)
    elif pooling_type == "avg":
        return GlobalMaskAvgPooling1D()
    elif pooling_type == "attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "context_attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "weighted_attention":
        return GlobalMaskWeightedAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "value_attention":
        return GlobalMaskValueAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "transformer":
        copy_config = copy.deepcopy(config)
        copy_config.hidden_size = hidden_size
        return GlobalMaskTransformerPooling1D(copy_config)
    else:
        return None


def create_activate(activate_func):
    if activate_func == "tanh":
        return nn.Tanh()
    elif activate_func == "relu":
        return nn.ReLU()
    elif activate_func == "leakyrelu":
        return nn.LeakyReLU()
    elif activate_func == "gelu":
        return nn.GELU()
    else:
        return nn.Tanh()


class SequenceAndStructureFusionNetwork(BertPreTrainedModel):
    '''
    DeepProtFunc
    '''
    def __init__(self, config, args=None):
        super(SequenceAndStructureFusionNetwork, self).__init__(config)
        self.num_labels = config.num_labels
        # sequence encoder, structure encoder, structural embedding encoder
        self.has_seq_encoder = args.has_seq_encoder
        self.has_struct_encoder = args.has_struct_encoder
        self.has_embedding_encoder = args.has_embedding_encoder
        assert args.has_seq_encoder or args.has_struct_encoder or args.has_embedding_encoder
        # includes sequence encoder
        if args.has_seq_encoder:
            # sequence -> transformer(k11 layers) + pooling + dense(k12 layers)
            self.seq_encoder = BertModel(config)
            self.seq_pooler = create_pooler(pooler_type="seq", config=config, args=args)
            assert isinstance(config.seq_fc_size, list)
            self.seq_linear = []
            input_size = config.hidden_size
            for idx in range(len(config.seq_fc_size)):
                linear = nn.Linear(input_size, config.seq_fc_size[idx])
                self.seq_linear.append(linear)
                self.seq_linear.append(create_activate(config.activate_func))
                input_size = config.seq_fc_size[idx]
            self.seq_linear = nn.ModuleList(self.seq_linear)
        # includes structure encoder
        if args.has_struct_encoder:
            # structure-> embedding + gcn(k21 layers) + pooling + dense(k22 layers)
            # k layers
            # output：[batch_size, seq_len, output_dim]
            self.struct_embedder = nn.Embedding(config.struct_vocab_size, config.struct_embed_size, padding_idx=config.pad_token_id)
            self.struct_encoder = []
            assert isinstance(config.struct_hidden_size, list) and isinstance(config.struct_output_size, list)
            input_size = config.struct_embed_size
            output_size = None
            assert len(config.struct_hidden_size) == len(config.struct_output_size)
            for idx in range(len(config.struct_output_size)):
                layer = GAT(feature_size=input_size,
                            hidden_size=config.struct_hidden_size[idx],
                            output_size=config.struct_output_size[idx],
                            dropout=config.hidden_dropout_prob,
                            nheads=config.struct_nb_heads,
                            alpha=config.struct_alpha)
                self.struct_encoder.append(layer)
                input_size = config.struct_output_size[idx]
                output_size = config.struct_output_size[idx]
            self.struct_encoder = nn.ModuleList(self.struct_encoder)
            self.struct_pooler = create_pooler(pooler_type="struct", config=config, args=args)
            assert isinstance(config.struct_fc_size, list)
            self.struct_linear = []
            input_size = output_size * len(config.struct_output_size)
            for idx in range(len(config.struct_fc_size)):
                linear = nn.Linear(input_size, config.struct_fc_size[idx])
                self.struct_linear.append(linear)
                self.struct_linear.append(create_activate(config.activate_func))
                input_size = config.struct_fc_size[idx]
            self.struct_linear = nn.ModuleList(self.struct_linear)
        # includes embedding encoder
        if args.has_embedding_encoder:
            self.embedding_pooler = create_pooler(pooler_type="embedding", config=config, args=args)
            assert isinstance(config.embedding_fc_size, list)
            self.embedding_linear = []
            input_size = config.embedding_input_size
            for idx in range(len(config.embedding_fc_size)):
                linear = nn.Linear(input_size, config.embedding_fc_size[idx])
                self.embedding_linear.append(linear)
                self.embedding_linear.append(create_activate(config.activate_func))
                input_size = config.embedding_fc_size[idx]
            self.embedding_linear = nn.ModuleList(self.embedding_linear)

        # weight assignment for addition of sequence, structure, structural embedding representation vector,
        # if none, concatenation, otherwise weighted sequence
        if args.has_seq_encoder and args.has_struct_encoder and args.has_embedding_encoder:
            if hasattr(config, "seq_weight") and hasattr(config, "struct_weight") and hasattr(config, "embedding_weight"):
                self.seq_weight = config.seq_weight
                self.struct_weight = config.struct_weight
                self.embedding_weight = config.embedding_weight
            else:
                self.seq_weight = None
                self.struct_weight = None
                self.embedding_weight = None
            assert self.seq_weight is None or self.seq_weight + self.struct_weight + self.embedding_weight == 1.0
            if self.seq_weight is None: # concat
                output_size = config.seq_fc_size[-1] + config.struct_fc_size[-1] + config.embedding_fc_size[-1]
            else: # add
                assert config.seq_fc_size[-1] == config.struct_fc_size[-1] == config.embedding_fc_size[-1]
                output_size = config.seq_fc_size[-1]
        elif args.has_seq_encoder and args.has_struct_encoder:
            if hasattr(config, "seq_weight") and hasattr(config, "struct_weight"):
                self.seq_weight = config.seq_weight
                self.struct_weight = config.struct_weight
            else:
                self.seq_weight = None
                self.struct_weight = None
            self.embedding_weight = None
            assert self.seq_weight is None or self.seq_weight + self.struct_weight == 1.0
            if self.seq_weight is None: # concat
                output_size = config.seq_fc_size[-1] + config.struct_fc_size[-1]
            else: # add
                assert config.seq_fc_size[-1] == config.struct_fc_size[-1]
                output_size = config.seq_fc_size[-1]
        elif args.has_seq_encoder and args.has_embedding_encoder:
            if hasattr(config, "seq_weight") and hasattr(config, "embedding_weight"):
                self.seq_weight = config.seq_weight
                self.embedding_weight = config.embedding_weight
            else:
                self.seq_weight = None
                self.embedding_weight = None
            self.struct_weight = None
            assert self.seq_weight is None or self.seq_weight + self.embedding_weight == 1.0
            if self.seq_weight is None: # concat
                output_size = config.seq_fc_size[-1] + config.embedding_fc_size[-1]
            else: # add
                assert config.seq_fc_size[-1] == config.embedding_fc_size[-1]
                output_size = config.seq_fc_size[-1]
        elif args.has_struct_encoder and args.has_embedding_encoder:
            if hasattr(config, "struct_weight") and hasattr(config, "embedding_weight"):
                self.struct_weight = config.struct_weight
                self.embedding_weight = config.embedding_weight
            else:
                self.struct_weight = None
                self.embedding_weight = None
            self.seq_weight = None
            assert self.struct_weight is None or self.struct_weight + self.embedding_weight == 1.0
            if self.struct_weight is None: # concat
                output_size = config.struct_fc_size[-1] + config.embedding_fc_size[-1]
            else: # add
                assert config.struct_fc_size[-1] == config.embedding_fc_size[-1]
                output_size = config.struct_fc_size[-1]
        else: # only one encoder
            self.seq_weight = None
            self.struct_weight = None
            self.embedding_weight = None
            output_size = config.seq_fc_size[-1] if args.has_seq_encoder else (config.struct_fc_size[-1] if args.has_struct_encoder else config.embedding_fc_size[-1])

        # dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # output layer
        self.output_mode = args.output_mode
        if args and args.sigmoid:
            if args.output_mode in ["binary_class", "binary-class"]:
                self.classifier = nn.Linear(output_size, 1)
            else:
                self.classifier = nn.Linear(output_size, config.num_labels)
            self.output = nn.Sigmoid()
        else:
            self.classifier = nn.Linear(output_size, config.num_labels)
            if self.num_labels > 1:
                self.output = nn.Softmax(dim=1)
            else:
                self.output = None

        # loss function type
        self.loss_type = args.loss_type

        # positive weight
        if hasattr(config, "pos_weight"):
            self.pos_weight = config.pos_weight
        elif hasattr(args, "pos_weight"):
            self.pos_weight = args.pos_weight
        else:
            self.pos_weight = None

        class_weights_input = None
        if hasattr(config, "weight") and config.weight is not None: # 优先从 config 获取
            class_weights_input = config.weight
            logger.info(f"Found 'weight' in config: {class_weights_input}")
        elif hasattr(args, "weight") and args.weight is not None: # 其次从 args 获取
            class_weights_input = args.weight
            logger.info(f"Found 'weight' in args: {class_weights_input}")
        else:
            logger.info("No 'weight' found in config or args.")

        self.weight = None # 初始化 self.weight 为 None

        if class_weights_input is not None:
            if isinstance(class_weights_input, str):
                try:
                    logger.info(f"Parsing class weights from string: '{class_weights_input}'")
                    weight_values = [float(v.strip()) for v in class_weights_input.split(',')]
                    # 将权重张量移动到正确的设备上，如果 args.device 可用
                    device_to_use = args.device if args and hasattr(args, 'device') else torch.device("cpu")
                    self.weight = torch.tensor(weight_values, dtype=torch.float).to(device_to_use)
                    logger.info(f"Parsed class weights tensor: {self.weight} on device {self.weight.device}")
                except ValueError as e:
                    logger.error(f"Error parsing weight string '{class_weights_input}': {e}. Expected comma-separated floats. Weight will be None.")
                    self.weight = None # 解析失败，则不使用权重
            elif isinstance(class_weights_input, (list, tuple)):
                logger.info(f"Converting class weights from list/tuple: {class_weights_input}")
                device_to_use = args.device if args and hasattr(args, 'device') else torch.device("cpu")
                self.weight = torch.tensor(class_weights_input, dtype=torch.float).to(device_to_use)
                logger.info(f"Converted class weights tensor: {self.weight} on device {self.weight.device}")
            elif torch.is_tensor(class_weights_input):
                logger.info(f"Using provided tensor for class weights: {class_weights_input}")
                device_to_use = args.device if args and hasattr(args, 'device') else torch.device("cpu")
                self.weight = class_weights_input.to(dtype=torch.float).to(device_to_use) # 确保是 float 类型并移动到设备
                logger.info(f"Class weights tensor: {self.weight} on device {self.weight.device}")
            else:
                logger.warning(f"Unsupported type for weight: {type(class_weights_input)}. Weight will be None.")
                self.weight = None
        else:
            self.weight = None # 确保在没有输入时 self.weight 是 None

        # self.pos_weight 的处理也可能需要类似逻辑，如果它也可能来自字符串输入
        # 但从你的代码看，pos_weight 似乎主要用于二分类，并且有更具体的类型转换
        # if self.pos_weight is not None and isinstance(self.pos_weight, str) and self.output_mode in ["multi_label"]:
        #     # ... 类似地解析 self.pos_weight ... (如果需要)e

        if self.output_mode in ["regression"]:
            self.loss_fct = MSELoss()
        elif self.output_mode in ["multi_label", "multi-label"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [1, 1, 1, ,1, 1...] length: self.num_labels
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == self.num_labels
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss(reduction=config.loss_reduction if hasattr(config, "loss_reduction") else "sum")
            elif self.loss_type == "asl":
                self.loss_fct = AsymmetricLossOptimized(gamma_neg=args.asl_gamma_neg if hasattr(args, "asl_gamma_neg") else 4,
                                                        gamma_pos=args.asl_gamma_pos if hasattr(args, "asl_gamma_pos") else 1,
                                                        clip=args.clip if hasattr(args, "clip") else 0.05,
                                                        eps=args.eps if hasattr(args, "eps") else 1e-8,
                                                        disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss if hasattr(args, "disable_torch_grad_focal_loss") else False)
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
            elif self.loss_type == "multilabel_cce":
                self.loss_fct = MultiLabel_CCE(normalization=False)
        elif self.output_mode in ["binary_class", "binary-class"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [0.9]
                    if isinstance(self.pos_weight, int):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.long).to(args.device)
                    elif isinstance(self.pos_weight, float):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32).to(args.device)

                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == 1
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss()
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
        elif self.output_mode in ["multi_class", "multi-class"]:
            if self.weight is not None: # <--- 修改这里：检查是否为 None
                # [1, 1, 1, ,1, 1...] length: self.num_labels
                assert self.weight.ndim == 1 and self.weight.shape[0] == self.num_labels, \
                    f"Weight tensor shape mismatch. Expected 1D tensor with {self.num_labels} elements, got shape {self.weight.shape} and value {self.weight}"
                self.loss_fct = CrossEntropyLoss(weight=self.weight)
                logger.info(f"Using CrossEntropyLoss with custom weights: {self.weight}")
            else:
                self.loss_fct = CrossEntropyLoss()
                logger.info("Using CrossEntropyLoss with default (no) weights.")
        else:
            raise Exception("Not support output mode: %s." % self.output_mode)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            struct_input_ids=None,
            struct_contact_map=None,
            embedding_info=None,
            embedding_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        seq_pooled_output = None
        if self.has_seq_encoder:
            # calc for sequence
            if input_ids is not None:
                seq_outputs = self.seq_encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                if self.seq_pooler is None:
                    # [CLS] vector
                    seq_pooled_output = seq_outputs[1]
                else:
                    # matrix to vector
                    seq_pooled_output = self.seq_pooler(seq_outputs[0])
                seq_pooled_output = self.dropout(seq_pooled_output)
                for i, layer_module in enumerate(self.seq_linear):
                    seq_pooled_output = layer_module(seq_pooled_output)
        struct_pooled_output = None
        if self.has_struct_encoder:
            if struct_input_ids is not None:
                assert struct_contact_map is not None
                struct_encoding_list = []
                hidden_states = self.struct_embedder(struct_input_ids)
                for i, layer_module in enumerate(self.struct_encoder):
                    hidden_states = layer_module(hidden_states, struct_contact_map)
                    struct_encoding_list.append(hidden_states)
                if len(struct_encoding_list) > 1:
                    hidden_states = torch.cat(struct_encoding_list, dim=-1)
                else:
                    hidden_states = struct_encoding_list[-1]
                struct_pooled_output = self.struct_pooler(hidden_states)
                struct_pooled_output = self.dropout(struct_pooled_output)
                for i, layer_module in enumerate(self.struct_linear):
                    struct_pooled_output = layer_module(struct_pooled_output)
        embedding_pooled_output = None
        if self.has_embedding_encoder:
            if self.embedding_pooler:
                embedding_pooled_output = self.embedding_pooler(embedding_info, mask=embedding_attention_mask)
            else:
                embedding_pooled_output = embedding_info

            embedding_pooled_output = self.dropout(embedding_pooled_output)
            for i, layer_module in enumerate(self.embedding_linear):
                embedding_pooled_output = layer_module(embedding_pooled_output)

        assert seq_pooled_output is not None or struct_pooled_output is not None or embedding_pooled_output is not None

        if seq_pooled_output is None and embedding_pooled_output is None:
            pooled_output = struct_pooled_output
        elif struct_pooled_output is None and embedding_pooled_output is None:
            pooled_output = seq_pooled_output
        elif seq_pooled_output is None and struct_pooled_output is None:
            pooled_output = embedding_pooled_output
        else:
            if self.seq_weight is not None and self.struct_weight is not None and self.embedding_weight is not None:
                pooled_output = torch.add(self.seq_weight * seq_pooled_output, self.struct_weight * struct_pooled_output, self.embedding_weight * embedding_pooled_output)
            elif self.seq_weight is not None and self.struct_weight is not None:
                pooled_output = torch.add(self.seq_weight * seq_pooled_output, self.struct_weight * struct_pooled_output)
            elif self.seq_weight is not None and self.embedding_weight is not None:
                pooled_output = torch.add(self.seq_weight * seq_pooled_output, self.embedding_weight * embedding_pooled_output)
            elif self.struct_weight is not None and self.embedding_weight is not None:
                pooled_output = torch.add(self.struct_weight * struct_pooled_output, self.embedding_weight * embedding_pooled_output)
            elif seq_pooled_output is not None and struct_pooled_output is not None and embedding_pooled_output is not None:
                pooled_output = torch.cat([seq_pooled_output, struct_pooled_output, embedding_pooled_output], dim=-1)
            elif seq_pooled_output is not None and struct_pooled_output is not None:
                pooled_output = torch.cat([seq_pooled_output, struct_pooled_output], dim=-1)
            elif seq_pooled_output is not None and embedding_pooled_output is not None:
                pooled_output = torch.cat([seq_pooled_output, embedding_pooled_output], dim=-1)
            elif struct_pooled_output is not None and embedding_pooled_output is not None:
                pooled_output = torch.cat([struct_pooled_output, embedding_pooled_output], dim=-1)
            else:
                raise Exception("Not support this type.")

        logits = self.classifier(pooled_output)
        if self.output:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]

        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception("Not support output_mode=%s" % self.output_mode)
            outputs = [loss, *outputs]

        return outputs




