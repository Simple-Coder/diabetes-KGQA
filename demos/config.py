"""
Created by xiedong
@Date: 2023/4/9 18:44
"""
import torch
import os


class Args:
    # 数据集类型：atis or snips
    task = 'atis'
    # 数据存放路径
    data_dir = '../intent_slot_fill_demo/data'

    # 意图标签文件名
    intent_label_file = 'intent_label.txt'
    # 槽位标签文件名
    slot_label_file = 'slot_label.txt'
    # bert模型
    bert_dir = 'bert-base-uncased'
    # 模型加载与保存的路径: atis_model 、snips_model
    model_dir = 'atis_model'

    do_train = True
    do_eval = False
    do_test = False
    do_save = True
    do_predict = True
    load_model = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 批处理大小
    train_batch_size = 32
    eval_batch_size = 64

    learning_rate = 5e-5
    num_train_epochs = 10
    dropout_rate = 0.1

    data_input_text_file = 'seq.in'
    data_intent_label_file = 'label'
    data_slot_labels_file = 'seq.out'

    # 意图编码
    intent_label_2id = {}
    id2_intent_label = {}
    with open(os.path.join(data_dir, task, intent_label_file), 'r') as fp:
        intent_labels = fp.read().split('\n')
        for i, intent in enumerate(intent_labels):
            intent_label_2id[intent] = i
            id2_intent_label[i] = intent


if __name__ == '__main__':
    args = Args()
    print(args.intent_label_2id)
    print(args.id2_intent_label)
