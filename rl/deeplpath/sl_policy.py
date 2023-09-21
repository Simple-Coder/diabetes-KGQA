import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from networks import PolicyNN
from utils import *
from env import Env
from BFS.KB import KB
from BFS.BFS import BFS
import time

relation = sys.argv[1]
# episodes = int(sys.argv[2])
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'


class SupervisedPolicy(nn.Module):
    def __init__(self, state_dim, action_space, learning_rate=0.001):
        super(SupervisedPolicy, self).__init__()

        self.state_dim = state_dim  # 状态维度
        self.action_space = action_space  # 动作空间的大小

        # 定义神经网络的层
        self.fc1 = nn.Linear(state_dim, 64)  # 调整隐藏层大小以满足需求
        self.fc2 = nn.Linear(64, action_space)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Adam优化器

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # 使用ReLU激活函数进行前向传播
        action_prob = self.fc2(x)
        return self.log_softmax(action_prob)  # 使用LogSoftmax获取动作概率的对数值

    def predict(self, state):
        with torch.no_grad():
            action_prob = self.forward(state)
        return action_prob.exp()  # 返回动作概率的指数值

    def update(self, state, action):
        self.optimizer.zero_grad()  # 梯度清零
        action_prob = self.forward(state)
        picked_action_prob = action_prob.gather(1, action.unsqueeze(1))  # 选择与实际动作对应的概率
        loss = -picked_action_prob.mean()  # 计算损失，采用负对数似然
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新模型参数
        return loss.item()  # 返回损失值
