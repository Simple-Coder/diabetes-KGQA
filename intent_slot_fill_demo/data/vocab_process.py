"""
Created by xiedong
@Date: 2023/4/9 15:19
"""
import os


def vocab_process(data_dir):
    # 槽位标签
    slot_label_vocab = 'slot_label.txt'
    # 意图标签
    intent_label_vocab = 'intent_label.txt'

    # 训练数据目录
    train_dir = os.path.join(data_dir, 'train')

    '''
    1、意图处理
    建立意图表字典
    '''
    # 原始意图表
    train_label_path = os.path.join(train_dir, 'label')
    # 处理后的意图set集合
    train_label_distinct_path = os.path.join(data_dir, intent_label_vocab)
    with open(train_label_path, 'r', encoding='utf-8') as f_r, open(train_label_distinct_path, 'w',
                                                                    encoding='utf-8') as f_w:
        # 声明意图定义集合
        intent_vocab = set()
        for line in f_r:
            line = line.strip()
            intent_vocab.add(line)

        # 添加未知意图
        additional_tokens = ["UNK"]
        for token in additional_tokens:
            f_w.write(token + "\n")

        # 意图排序
        intent_vocab = sorted(list(intent_vocab))
        for intent in intent_vocab:
            f_w.write(intent + "\n")

    '''
    2、槽位处理
    建立槽位表字典
    '''
    # 原始槽位表
    train_slot_path = os.path.join(train_dir, 'seq.out')
    # 处理后的槽位表
    train_slot_distinct_path = os.path.join(data_dir, slot_label_vocab)
    with open(train_slot_path, 'r', encoding='utf-8') as f_r, open(train_slot_distinct_path, 'w',
                                                                   encoding='utf-8') as f_w:
        # 声明槽位集合
        slot_vocab = set()

        for line in f_r:
            line = line.strip()
            # 按空格切分
            slots = line.split()
            for slot in slots:
                slot_vocab.add(slot)

        # 意图处理
        slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[2:]))

        # 写入文件
        additional_tokens = ["PAD", "UNK"]
        for token in additional_tokens:
            f_w.write(token + "\n")

        for slot in slot_vocab:
            f_w.write(slot + "\n")


if __name__ == '__main__':
    vocab_process('atis')
    vocab_process('snips')
