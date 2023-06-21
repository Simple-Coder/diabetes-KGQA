"""
Created by xiedong
@Date: 2023/6/21 16:52
"""
from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_model import MultiJointModel
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot


def vision_torchviz(args):
    # import os
    # os.environ["PATH"] += os.pathsep + 'D:/Softs/Graphviz 2.44.1/bin/'
    # 创建模型的实例
    num_intents = args.seq_num_labels  # 替换为实际的意图数目
    num_slots = args.token_num_labels  # 替换为实际的槽位数目
    model = MultiJointModel(num_intents, num_slots)

    # 创建示例输入张量
    batch_size = 10
    sequence_length = 20
    input_ids = torch.randint(0, 100, (batch_size, sequence_length))
    attention_mask = torch.ones((batch_size, sequence_length))

    # 生成模型的可视化
    output = model(input_ids, attention_mask)
    dot = make_dot(output, params=dict(model.named_parameters()))
    # dot = make_dot(output, params=dict(model.named_parameters()), graphviz_executables="D:/Softs/Graphviz 2.44.1/bin")
    dot.format = 'png'
    dot.render(filename="multi_joint_model", directory="./", format="png")  # 将可视化保存为PNG图像文件


def tensor_board_vision(args):
    # 创建模型的实例
    num_intents = args.seq_num_labels  # 替换为实际的意图数目
    num_slots = args.token_num_labels  # 替换为实际的槽位数目
    model = MultiJointModel(num_intents, num_slots)

    # 创建示例输入张量
    batch_size = 10
    sequence_length = 20
    input_ids = torch.randint(0, 100, (batch_size, sequence_length))
    attention_mask = torch.ones((batch_size, sequence_length))

    # 创建 TensorBoard 的 SummaryWriter
    writer = SummaryWriter()

    # 将模型的图形可视化写入 TensorBoard
    writer.add_graph(model, (input_ids, attention_mask))

    # 关闭 SummaryWriter
    writer.close()

if __name__ == '__main__':
    args = ModelConfig()
    # vision_torchviz(args)

    tensor_board_vision(args)
