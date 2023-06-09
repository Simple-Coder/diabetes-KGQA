"""
Created by xiedong
@Date: 2023/5/27 15:56
"""
import torch


class Args:
    # bert预训练模型
    bert_dir = 'hfl/chinese-bert-wwm-ext'
    # 模型保存位置
    save_dir = 'checkpoints/'
    # 加载模型位置
    # load_dir = '/Users/xiedong/PycharmProjects/diabetes-KGQA/server/build_model/checkpoints/1685456332_19_muti_model.pt'
    load_dir = 'D:\dev\PycharmProjects\diabetes-KGQA\server//build_model\muti_intent\checkpoints//1685507169_0_muti_model.pt'
    # load_dir = 'checkpoints/1682428820model.pt'

    # # 意图labels
    # seq_labels_path = '/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/intent_label.txt'
    # # 槽位labels
    # token_labels_path = '/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/slot_label.txt'
    #
    # train_texts = '/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/test/seq.in'
    # train_intents = '/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/test/label'
    # train_slots = '/Users/xiedong/PycharmProjects/diabetes-KGQA/server/data/intent_and_slot_data/test/seq.out'

    # 意图labels
    seq_labels_path = 'D://dev//PycharmProjects//diabetes-KGQA//server//data//intent_and_slot_data//intent_label.txt'
    # 槽位labels
    token_labels_path = 'D://dev//PycharmProjects//diabetes-KGQA//server//data//intent_and_slot_data//slot_label.txt'

    train_texts = 'D://dev//PycharmProjects//diabetes-KGQA//server//data//intent_and_slot_data//test/seq.in'
    train_intents = 'D://dev//PycharmProjects//diabetes-KGQA//server//data//intent_and_slot_data//test/label'
    train_slots = 'D://dev//PycharmProjects//diabetes-KGQA//server//data//intent_and_slot_data//test/seq.out'

    load_model = False
    do_train = True
    do_eval = False
    do_test = True
    do_save = True
    do_predict = True
    use_crf = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    # 意图数量
    seq_num_labels = len(seq_labels)
    # 槽位数量
    token_num_labels = len(token_labels)

    # 多标签分类意图阈值
    muti_intent_threshold = 0.5
    muti_intent_threshold_num = 2
    # 隐层大小
    hidden_size = 768
    # 最大支持长度
    max_len = 32
    # 批处理大小
    batchsize = 2
    # 学习率
    lr = 2e-5
    # 训练轮数
    epoch = 50
    # dropout率
    hidden_dropout_prob = 0.1


if __name__ == '__main__':
    args = Args()
    print('意图：', args.seq_labels)
    print('意图2：', args.seqlabel2id)
    print(args.tokenlabel2id)
    print(args.token_labels)

    # Define example data
    train_sentences = ['预订从上海到北京的机票', '提醒我明天上午8点开会']
    intent_labels = [['预订', '查询'], ['提醒', '安排会议']]

    # Create a list of all unique intents
    all_intents = sorted(list(set([intent for intents in intent_labels for intent in intents])))

    # Create a dictionary to map intents to indices
    intent_to_index = {intent: index for index, intent in enumerate(all_intents)}

    # Convert intent labels to one-hot encoding
    train_intent_labels = []
    for intents in intent_labels:
        intent_indices = [intent_to_index[intent] for intent in intents]
        one_hot = torch.zeros(len(all_intents))
        one_hot[intent_indices] = 1
        train_intent_labels.append(one_hot)

    train_intent_labels = torch.stack(train_intent_labels)

    print(train_intent_labels)
