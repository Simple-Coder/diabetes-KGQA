"""
Created by xiedong
@Date: 2023/4/20 15:25
"""
import torch.nn as nn
from transformers import BertModel
import torch


class BertForIntentClassificationAndSlotFilling(nn.Module):
    def __init__(self, config):
        super(BertForIntentClassificationAndSlotFilling, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_dir)
        self.bert_config = self.bert.config
        self.sequence_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.seq_num_labels),
        )
        self.token_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.token_num_labels),
        )
        # 增加sigmoid实现多标签分类
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                ):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        # print("bertOutput", bert_output)
        pooler_output = bert_output[1]  # CLS
        token_output = bert_output[0]
        seq_output = self.sequence_classification(pooler_output)
        token_output = self.token_classification(token_output)
        return seq_output, token_output


# 定义模型结构
class SlotGatedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SlotGatedModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.slot_gate = nn.Linear(hidden_dim * 2, 1)
        self.intent_classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, inputs):
        encoder_outputs, _ = self.encoder(inputs)
        slot_gate_weights = torch.sigmoid(self.slot_gate(encoder_outputs))
        slot_outputs = encoder_outputs * slot_gate_weights
        intent_outputs = torch.mean(encoder_outputs, dim=0)
        intent_outputs = torch.squeeze(intent_outputs, dim=0)
        intent_outputs = self.intent_classifier(intent_outputs)
        return slot_outputs, intent_outputs
