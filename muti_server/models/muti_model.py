"""
Created by xiedong
@Date: 2023/5/27 16:16
"""
import torch.nn as nn
from transformers import BertModel
from muti_server.models.muti_config import Args
from torchcrf import CRF
from transformers import logging

logging.set_verbosity_error()


class MutiJointModel(nn.Module):
    def __init__(self, num_intents, num_slots, hidden_dropout_prob=0.1):
        super(MutiJointModel, self).__init__()
        args = Args()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_filler = nn.Linear(self.bert.config.hidden_size, num_slots)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # self.intent_classifier = nn.Sequential(
        #     nn.Dropout(hidden_dropout_prob),
        #     nn.Linear(self.bert.config.hidden_size, num_intents),
        # )
        # self.slot_filler = nn.Sequential(
        #     nn.Dropout(hidden_dropout_prob),
        #     nn.Linear(self.bert.config.hidden_size, num_slots),
        # )
        # 增加sigmoid实现多标签分类
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

        # self.crf = CRF(num_tags=num_slots, batch_first=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 意图分类
        intent_logits = self.intent_classifier(pooled_output)
        intent_probs = self.sigmoid(intent_logits)

        # slot标注结果
        slot_logits = self.slot_filler(outputs.last_hidden_state)
        slot_probs = self.softmax(slot_logits)

        # return intent_logits, slot_logits
        return intent_probs, slot_probs
