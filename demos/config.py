"""
Created by xiedong
@Date: 2023/4/9 18:44
"""


class Args:
    # 数据集类型：atis or snips
    task = 'atis'
    # 模型加载与保存的路径: atis_model 、snips_model
    model_dir = 'atis_model'
    # 数据存放路径
    data_dir = '../intent_slot_fill_demo/data'
    # 意图标签文件名
    intent_label_file = 'intent_label.txt'
    # 槽位标签文件名
    slot_label_file = 'slot_label.txt'

    # 批处理大小
    train_batch_size = 32
    eval_batch_size = 64

    learning_rate = 5e-5
    num_train_epochs = 10
    dropout_rate = 0.1
