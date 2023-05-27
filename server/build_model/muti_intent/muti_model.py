"""
Created by xiedong
@Date: 2023/5/27 16:16
"""
import torch.nn as nn
from transformers import BertModel
from muti_config import Args


class BertForIntentClassificationAndSlotFilling(nn.Module):
    def __init__(self, num_intents, num_slots):
        super(BertForIntentClassificationAndSlotFilling, self).__init__()
        args = Args()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_filler = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # 意图分类
        intent_logits = self.intent_classifier(pooled_output)
        # slot标注结果
        slot_logits = self.slot_filler(outputs.last_hidden_state)

        return intent_logits, slot_logits
