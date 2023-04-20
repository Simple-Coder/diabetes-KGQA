"""
Created by xiedong
@Date: 2023/4/19 22:12
"""


class Args:
    # bert预训练模型
    bert_dir = 'hfl/chinese-bert-wwm-ext'
    # 模型保存位置
    save_dir = './checkpoints/'
    # 加载模型位置
    load_dir = './checkpoints/model.pt'

    # 意图labels
    seq_labels_path = '../data/intent_and_slot_data/intent_label.txt'
    # 槽位labels
    token_labels_path = '../data/intent_and_slot_data/slot_label.txt'

    load_model = False


    seqlabel2id = {}
    id2seqlabel = {}
    with open(seq_labels_path, 'r', encoding='utf-8') as fp:
        seq_labels = fp.read().split('\n')
        seq_labels = list(filter(lambda x: x != '', seq_labels))
        for i, label in enumerate(seq_labels):
            if label == '':
                continue
            seqlabel2id[label] = i
            id2seqlabel[i] = label

    tokenlabel2id = {}
    id2tokenlabel = {}
    with open(token_labels_path, 'r', encoding='utf-8') as fp:
        token_labels = fp.read().split('\n')
        token_labels = list(filter(lambda x: x != '', token_labels))
        for i, label in enumerate(token_labels):
            if label == '':
                continue
            tokenlabel2id[label] = i
            id2tokenlabel[i] = label

    # 隐层大小
    hidden_size = 768
    # 最大支持长度
    max_len = 32
    # 批处理大小
    batchsize = 64
    # 学习率
    lr = 2e-5
    # 训练轮数
    epoch = 10
    # dropout率
    hidden_dropout_prob = 0.1


if __name__ == '__main__':
    args = Args()
    print(args.seq_labels)
    print(args.seqlabel2id)
    print(args.tokenlabel2id)
    print(args.token_labels)
