"""
Created by xiedong
@Date: 2023/4/6 21:55
"""
import os


def vocab_process(data_dir, data_type):
    slot_label_vocab = 'slot_label.txt'
    intent_label_vocab = 'intent_label.txt'

    train_dir = os.path.join(data_dir, data_type)
    # intent
    with open(os.path.join(train_dir, 'label'), 'r', encoding='utf-8') as f_r, open(
            os.path.join(data_dir, intent_label_vocab), 'w',
            encoding='utf-8') as f_w:
        intent_vocab = set()
        for line in f_r:
            line = line.strip()
            intents = line.split('|')
            for intent in intents:
                intent_vocab.add(intent)

        additional_tokens = ["others"]
        for token in additional_tokens:
            f_w.write(token + '\n')

        intent_vocab = sorted(list(intent_vocab))
        for intent in intent_vocab:
            f_w.write(intent + '\n')

    # slot
    with open(os.path.join(train_dir, 'seq.out'), 'r', encoding='utf-8') as f_r, open(
            os.path.join(data_dir, slot_label_vocab), 'w',
            encoding='utf-8') as f_w:
        slot_vocab = set()
        for line in f_r:
            line = line.strip()
            slots = line.split()
            for slot in slots:
                slot_vocab.add(slot)

        slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))

        # Write additional tokens
        additional_tokens = ["PAD", "UNK"]
        for token in additional_tokens:
            f_w.write(token + '\n')

        for slot in slot_vocab:
            f_w.write(slot + '\n')


if __name__ == '__main__':
    vocab_process('./', 'test')
    # vocab_process('./','train')
