"""
Created by xiedong
@Date: 2023/4/9 16:48
"""
import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from intent_slot_fill_demo.model.model_jointbert import JointBERT

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer)
}
MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased'
}


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def get_intent_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]
