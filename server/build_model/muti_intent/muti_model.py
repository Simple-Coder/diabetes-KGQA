"""
Created by xiedong
@Date: 2023/5/27 16:16
"""
import torch.nn as nn
from transformers import BertModel
from muti_config import Args


class MutiJointModel(nn.Module):
    def __init__(self, num_intents, num_slots, hidden_dropout_prob=0.1):
        super(MutiJointModel, self).__init__()
        args = Args()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        # self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        # self.slot_filler = nn.Linear(self.bert.config.hidden_size, num_slots)

        self.intent_classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, num_intents),
        )
        self.slot_filler = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, num_slots),
        )
        # 增加sigmoid实现多标签分类
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # 意图分类
        intent_logits = self.intent_classifier(pooled_output)

        intent_logits_sigmoid = self.sigmoid(intent_logits)

        # slot标注结果
        slot_logits = self.slot_filler(outputs.last_hidden_state)

        # return intent_logits, slot_logits
        return intent_logits_sigmoid, slot_logits
