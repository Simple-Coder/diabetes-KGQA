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
