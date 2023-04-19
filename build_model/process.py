"""
Created by xiedong
@Date: 2023/4/19 22:17
"""
import torch
import re


class InputExample:
    def __init__(self, set_type, text, seq_label, token_label):
        self.set_type = set_type
        self.text = text
        self.seq_label = seq_label
        self.token_label = token_label


class InputFeature:
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 seq_label_ids,
                 token_label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.seq_label_ids = seq_label_ids
        self.token_label_ids = token_label_ids

