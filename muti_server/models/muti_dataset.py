"""
Created by xiedong
@Date: 2023/5/27 16:25
"""
from torch.utils import data
from torch.utils.data import Dataset
from transformers import logging

logging.set_verbosity_error()

from muti_server.models.muti_config import ModelConfig
from transformers import BertTokenizer
from muti_process import Processor, get_features
import torch


class BertDataset(data.Dataset):
    def __init__(self, type='train'):
        args = ModelConfig()
        # self.features = features
        # self.nums = len(self.features)
        tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

        raw_examples = Processor.get_examples(Processor.read_file(args.train_texts),
                                              Processor.read_file(args.train_intents),
                                              Processor.read_file(args.train_slots), 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        self.features = train_features
        self.nums = len(self.features)

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            'input_ids': self.features[item].input_ids.long(),
            'attention_mask': self.features[item].attention_mask.long(),
            'token_type_ids': self.features[item].token_type_ids.long(),
            'seq_label_ids': self.features[item].seq_label_ids.long(),
            'token_label_ids': self.features[item].token_label_ids.long(),
        }
        return data


# 定义数据集类
class SLUDataset(Dataset):
    def __init__(self):
        args = ModelConfig()
        # self.features = features
        # self.nums = len(self.features)
        tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

        raw_examples = Processor.get_examples(Processor.read_file(args.train_texts),
                                              Processor.read_file(args.train_intents),
                                              Processor.read_file(args.train_slots), 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        self.features = train_features
        self.nums = len(self.features)

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            'input_ids': self.features[item].input_ids.long(),
            'attention_mask': self.features[item].attention_mask.long(),
            'token_type_ids': self.features[item].token_type_ids.long(),
            'seq_label_ids': self.features[item].seq_label_ids.long(),
            'token_label_ids': self.features[item].token_label_ids.long(),
        }
        # return input_seq, intent_label, slot_labels
        return data
