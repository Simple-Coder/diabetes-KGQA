"""
Created by xiedong
@Date: 2023/8/5 21:00
"""
from muti_server.models.muti_config import ModelConfig
from transformers import BertTokenizer, BertModel

config = ModelConfig()

tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
model = BertModel.from_pretrained(config.bert_dir)
