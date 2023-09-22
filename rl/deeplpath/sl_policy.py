import sys
from itertools import count

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

        self.policy_nn = PolicyNN(state_dim, action_space)
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()  # 使用负对数似然损失

    def forward(self, state):
        return self.policy_nn(state)

    def predict(self, state):
        with torch.no_grad():
            action_prob = self.forward(state)
            return action_prob.exp().numpy()

    def update(self, state, action):
        self.optimizer.zero_grad()
        action_prob = self.forward(state)
        # 将 action 调整为二维张量以匹配输入
        action = action.view(-1, 1)
        loss = self.criterion(action_prob, action.squeeze())
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == '__main__':
    # 示例用法:
    state_dim = 4  # 替换为实际状态维度
    action_space = 2  # 替换为实际动作空间大小
    policy = SupervisedPolicy(state_dim, action_space)

    # 更新策略
    sample_state = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    sample_action = torch.tensor([0], dtype=torch.int64)  # 假设只有一个动作
    loss = policy.update(sample_state, sample_action)

    print(loss)