"""
Created by xiedong
@Date: 2023/6/9 23:36
"""
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialog System")

    parser.add_argument("--bert_dir", type=str, default="hfl/chinese-bert-wwm-ext", help="bert预训练模型")
    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="模型保存位置")
    parser.add_argument("--load_dir", type=str,
                        default="/Users/xiedong/PycharmProjects/diabetes-KGQA/server/build_model/checkpoints/1685456332_19_muti_model.pt",
                        help="加载模型位置")
    parser.add_argument("--seq_labels_path", type=str,
                        default="/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/intent_label.txt",
                        help="意图labels")
    parser.add_argument("--token_labels_path", type=str,
                        default="/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/slot_label.txt",
                        help="意图labels")
    parser.add_argument("--train_texts", type=str,
                        default="/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/test/seq.in",
                        help="train_texts")

    parser.add_argument("--train_intents", type=str,
                        default="/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/test/label",
                        help="train_intents")

    parser.add_argument("--train_slots", type=str,
                        default="/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/test/seq.out",
                        help="train_slots")

    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="模型保存位置")

    return parser.parse_args()
