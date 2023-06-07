"""
Created by xiedong
@Date: 2023/5/27 16:03
"""
import jieba
import torch
import re

from muti_server.models.muti_config import Args
from transformers import BertTokenizer
from transformers import logging

logging.set_verbosity_error()


class MutiInputExample:
    def __init__(self, set_type, text, seq_label, token_label):
        self.set_type = set_type
        self.text = text
        self.seq_label = seq_label
        self.token_label = token_label


class MutiInputFeature:
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


class Processor:
    @classmethod
    def get_examples(cls, texts, intents, slots, set_type):
        raw_examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            raw_examples.append(
                MutiInputExample(
                    set_type,
                    text,
                    intent,
                    slot
                )
            )
        return raw_examples

    @classmethod
    def read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines


def convert_example_to_feature(ex_idx, example, tokenizer, config):
    set_type = example.set_type
    text = example.text
    seq_label = example.seq_label
    token_label = example.token_label

    # 1、处理多标签分类'method_drug'
    intput_intent_list = seq_label.split('|')
    intent_indices = [config.seqlabel2id[intent] for intent in intput_intent_list]
    seq_label_ids = torch.zeros(config.seq_num_labels)
    seq_label_ids[intent_indices] = 1
    # seq_label_ids = config.seqlabel2id[seq_label]

    token_label_ids = []
    for s in token_label.split():
        # slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
        token_label_ids.append(
            config.token_labels.index(s) if s in config.token_labels else config.token_labels.index("UNK"))

    if len(token_label_ids) >= config.max_len - 2:
        token_label_ids = [0] + token_label_ids[0:config.max_len - 2] + [0]
    else:
        token_label_ids = [0] + token_label_ids + [0] + [0] * (config.max_len - len(token_label_ids) - 2)
    # print(token_label_ids)

    text = get_word_list(text)
    # text = list(jieba.cut(text))
    # text = [i for i in text]
    inputs = tokenizer.encode_plus(
        text=text,
        max_length=config.max_len,
        padding='max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    # input_ids = torch.tensor(inputs['input_ids'], requires_grad=False)
    # attention_mask = torch.tensor(inputs['attention_mask'], requires_grad=False)
    # token_type_ids = torch.tensor(inputs['token_type_ids'], requires_grad=False)
    # seq_label_ids = torch.tensor(seq_label_ids, requires_grad=False)
    # token_label_ids = torch.tensor(token_label_ids, requires_grad=False)

    input_ids = torch.tensor(inputs['input_ids'])
    attention_mask = torch.tensor(inputs['attention_mask'])
    token_type_ids = torch.tensor(inputs['token_type_ids'])
    seq_label_ids = torch.tensor(seq_label_ids)
    token_label_ids = torch.tensor(token_label_ids)

    if ex_idx < 3:
        print(f'*** {set_type}_example-{ex_idx} ***')
        print(f'text: {text}')
        print(f'input_ids: {input_ids}')
        print(f'attention_mask: {attention_mask}')
        print(f'token_type_ids: {token_type_ids}')
        print(f'seq_label_ids: {seq_label_ids}')
        print(f'token_label_ids: {token_label_ids}')

    feature = MutiInputFeature(
        input_ids,
        attention_mask,
        token_type_ids,
        seq_label_ids,
        token_label_ids,
    )

    return feature


def get_features(raw_examples, tokenizer, args):
    features = []
    for i, example in enumerate(raw_examples):
        feature = convert_example_to_feature(i, example, tokenizer, args)
        features.append(feature)
    return features


def get_word_list(s1):
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    # regEx = re.compile('[\\W]*')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    regEx = re.compile('\W+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r"([\u4e00-\u9fa5])")  # [\u4e00-\u9fa5]中文范围

    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符

    return list_word1


if __name__ == '__main__':
    s = "12、China's Legend Holdings will split its several business arms to go public on stock markets,  the group's president Zhu Linan said on Tuesday.该集团总裁朱利安周二表示，haha中国联想控股将分拆其多个业务部门在股市上市。"
    list_word1 = get_word_list(s)
    print(list_word1)
    list_word1 = list(jieba.cut(s))
    print(list_word1)

    args = Args()

    texts = Processor.read_file('../../data/intent_and_slot_data/test/seq.in')
    intents = Processor.read_file('../../data/intent_and_slot_data/test/label')
    slots = Processor.read_file('../../data/intent_and_slot_data/test/seq.out')

    raw_examples = Processor.get_examples(texts, intents, slots, 'train')
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    features = get_features(raw_examples, tokenizer, args)
    print("")
