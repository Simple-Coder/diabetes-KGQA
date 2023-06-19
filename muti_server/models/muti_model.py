"""
Created by xiedong
@Date: 2023/5/27 16:16
"""
import torch.nn as nn
from transformers import BertModel
from muti_server.models.muti_config import ModelConfig
from torchcrf import CRF
from transformers import logging

logging.set_verbosity_error()


class MultiJointModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super(MultiJointModel, self).__init__()
        model_config = ModelConfig()
        self.bert = BertModel.from_pretrained(model_config.bert_dir)
        self.intent_classifier = nn.Sequential(
            nn.Dropout(model_config.hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, num_intents),
            nn.Sigmoid()  # 对多标签分类应用Sigmoid激活函数
        )
        self.slot_filler = nn.Sequential(
            nn.Dropout(model_config.hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, num_slots),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs[1]
        token_output = outputs[0]

        # 意图分类
        intent_output = self.intent_classifier(pooler_output)

        # Slot填充
        slot_output = self.slot_filler(token_output)

        return intent_output, slot_output

    def calculate_loss(self, intent_output, slot_output, intent_labels, slot_labels):
        intent_loss = nn.BCELoss()(intent_output, intent_labels.float())
        slot_loss = nn.CrossEntropyLoss()(slot_output.view(-1, slot_output.size(-1)), slot_labels.view(-1))
        total_loss = intent_loss + slot_loss
        return total_loss

# class MutiJointModel(nn.Module):
#     def __init__(self, num_intents, num_slots, hidden_dropout_prob=0.1):
#         super(MutiJointModel, self).__init__()
#         model_config = ModelConfig()
#         self.bert = BertModel.from_pretrained(model_config.bert_dir)
#         self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
#         self.slot_filler = nn.Linear(self.bert.config.hidden_size, num_slots)
#         self.dropout = nn.Dropout(hidden_dropout_prob)
#         # self.intent_classifier = nn.Sequential(
#         #     nn.Dropout(hidden_dropout_prob),
#         #     nn.Linear(self.bert.config.hidden_size, num_intents),
#         # )
#         # self.slot_filler = nn.Sequential(
#         #     nn.Dropout(hidden_dropout_prob),
#         #     nn.Linear(self.bert.config.hidden_size, num_slots),
#         # )
#         # 增加sigmoid实现多标签分类
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=2)
#
#         self.crf = CRF(num_tags=num_slots, batch_first=True)
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#
#         # 意图分类
#         intent_logits = self.intent_classifier(pooled_output)
#         intent_probs = self.sigmoid(intent_logits)
#
#         # slot标注结果
#         slot_logits = self.slot_filler(outputs.last_hidden_state)
#         slot_probs = self.softmax(slot_logits)
#
#         # return intent_logits, slot_logits
#         return intent_probs, slot_probs


# 定义 JointSLU 模型
# class JointSLU(nn.Module):
#     def __init__(self, num_intents, num_slots, embedding_dim, hidden_dim, num_embeddings):
#         super(JointSLU, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
#         self.intent_classifier = nn.Linear(hidden_dim * 2, num_intents)
#         self.slot_tagger = nn.Linear(hidden_dim * 2, num_slots)
#
#     def forward(self, input_seq):
#         embedded = self.embedding(input_seq)
#         lstm_out, _ = self.lstm(embedded)
#         intent_logits = self.intent_classifier(lstm_out[:, -1, :])
#         slot_logits = self.slot_tagger(lstm_out)
#         return intent_logits, slot_logits
# TODO:其他模型
