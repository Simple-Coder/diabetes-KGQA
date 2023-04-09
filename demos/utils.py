"""
Created by xiedong
@Date: 2023/4/9 19:00
"""
import logging
import os


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


"""
获取意图数据集
"""


def get_intent_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


"""
获取槽位数据集
"""


def get_slot_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


if __name__ == '__main__':
    from config import Args

    args = Args()
    labels = get_intent_labels(args)
    print(labels)
    slots = get_slot_labels(args)
    print(slots)
