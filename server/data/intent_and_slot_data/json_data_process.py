"""
Created by xiedong
@Date: 2023/5/9 15:35
"""
import json
import random

"""
Created by xiedong
@Date: 2023/4/26 12:42
"""
from tqdm import tqdm

import re


class InputExample:
    def __init__(self, text, seq_label, token_label):
        self.text = text
        self.seq_label = seq_label
        self.token_label = token_label


class TodoDataProcessor:
    def __init__(self, intent_path, text_path, slot_path, todo_data_path):
        self.intent_path = intent_path
        self.text_path = text_path
        self.slot_path = slot_path
        self.todo_data_path = todo_data_path

    def get_examples(self):
        raw_examples = []
        with open(self.todo_data_path, 'r', encoding='utf-8') as f:
            data = eval(f.read())
            for i, d in tqdm(enumerate(data)):
                text = d['text']
                seq_label = d['intent']
                token_label = d['slots']
                raw_examples.append(
                    InputExample(
                        text,
                        seq_label,
                        token_label
                    )
                )
            return raw_examples

    def write_example_to_data(self, ex_idx, example):
        text = example.text
        seq_label = example.seq_label
        token_label = example.token_label

        # slot使用
        texts = get_word_list(text)
        slots = ['O'] * len(texts)
        for k, v in token_label.items():
            # print(k, v, text)
            re_res = re.finditer(v, text)
            for span in re_res:
                entity = span.group()
                start = span.start()
                end = span.end()
                # print(entity, start, end)
                slots[start] = 'B-' + k
                for i in range(start + 1, end):
                    slots[i] = 'I-' + k

        slotsStr = ' '.join('{0}'.format(x) for x in slots)

        if ex_idx < 3:
            print(f'*** example-{ex_idx} ***')
            print(f'text: {text}')
            print(f'intent: {seq_label}')
            print(f'slots: {slotsStr}')

        with open(self.intent_path, 'a', encoding='utf-8') as f_intent:
            f_intent.write(seq_label + '\n')

        with open(self.text_path, 'a', encoding='utf-8') as f_text:
            f_text.write(text + '\n')

        with open(self.slot_path, 'a', encoding='utf-8') as f_slot:
            f_slot.write(slotsStr + '\n')


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

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符#

    return list_word1


def train_test_split(all_tata_path, ratio=0.9):
    with open(all_tata_path, 'r') as fp:
        data = eval(fp.read())
        random.shuffle(data)
        total = len(data)
        train_data = data[:int(total * ratio)]
        test_data = data[int(total * ratio):]
    with open('train_process.json', 'w') as fp:
        json.dump(train_data, fp, ensure_ascii=False)
    with open('test_process.json', 'w') as fp:
        json.dump(test_data, fp, ensure_ascii=False)


def build_data(todo_data_path, intent_path, text_path, slot_path):
    # 处理抓取数据 ####
    processor = TodoDataProcessor(intent_path, text_path, slot_path, todo_data_path)

    examples = processor.get_examples()

    for i, example in tqdm(enumerate(examples)):
        try:
            processor.write_example_to_data(i, example)
        except Exception as e:
            print("错误行")
            print(i)
            print(example.__dict__)
            print(e)
            print(e)
    print()


if __name__ == '__main__':
    # train_test_split('./origin2.json') ##

    intent_path_train = './train/label'
    text_path_train = './train/seq.in'
    slot_path_train = './train/seq.out'
    todo_data_path_train = 'origin2.json'
    # todo_data_path_train = 'origin.json'

    build_data(todo_data_path_train, intent_path_train, text_path_train, slot_path_train)

    intent_path_test = './test/label'
    text_path_test = './test/seq.in'
    slot_path_test = './test/seq.out'
    # todo_data_path_test = 'origin.json'
    todo_data_path_test = 'origin2.json'

    build_data(todo_data_path_test, intent_path_test, text_path_test, slot_path_test)


    # intent_path = './test/label'
    # text_path = './test/seq.in'
    # slot_path = './test/seq.out'

    # intent_path = './train/label'
    # text_path = './train/seq.in'
    # slot_path = './train/seq.out'
    # todo_data_path = 'origin.json'

    # intent_path = './test/label'
    # text_path = './test/seq.in'
    # slot_path = './test/seq.out'
    # todo_data_path = 'origin2.json'
