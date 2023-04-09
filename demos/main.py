"""
Created by xiedong
@Date: 2023/4/9 18:42
"""
from config import Args
from transformers import BertTokenizer
from utils import *

if __name__ == '__main__':
    # 日志记录
    init_logger()
    # 参数配置
    args = Args()
    # 加载预训练模型
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
