import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import time

from sklearn.metrics.pairwise import cosine_similarity
from itertools import count

from networks import PolicyNN, ValueNN  # 假设你有适用于PyTorch的网络模型
from utils import *
from env import Env

# 获取命令行参数
relation = sys.argv[1]
task = sys.argv[2]
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_space, learning_rate=0.001):
        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        # 创建策略网络
        self.policy_nn = PolicyNN(state_dim, action_space)
        # 定义优化器
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=learning_rate)

    def forward(self, state):
        action_prob = self.policy_nn(state)
        return action_prob

    def compute_loss(self, action_prob, target, action):
        # TODO: Add regularization loss
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob) * target)
        return loss


def train_test():
    # 定义训练参数
    num_episodes = 5
    state_dim = 4
    action_space = 2
    learning_rate = 0.001

    # 创建策略网络
    policy_net = PolicyNetwork(state_dim, action_space, learning_rate)

    for episode in range(num_episodes):
        print(f"第 {episode + 1} 轮训练：")

        # 模拟一次轮回
        for step in range(100):  # 假设每轮有100个步骤
            # 生成示例状态
            state = np.random.rand(state_dim).astype(np.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32)  # 将状态转换为PyTorch张量

            # 使用策略网络预测动作概率
            action_prob = policy_net(state_tensor)

            # 根据概率选择动作
            # 根据概率选择动作
            action = np.random.choice(action_space, p=action_prob.detach().numpy())
            action_tensor = torch.tensor(action, dtype=torch.int64)  # 将action转换为PyTorch张量

            # 模拟环境，生成示例奖励
            reward = random.uniform(0, 1)

            # 计算损失并更新策略网络
            loss = policy_net.compute_loss(action_prob, reward, action_tensor)
            policy_net.optimizer.zero_grad()
            loss.backward()
            policy_net.optimizer.step()

            # 打印训练信息
            print(f"步骤 {step + 1}: 状态={state}, 动作={action}, 奖励={reward:.2f}")

    print("训练完成！")

if __name__ == "__main__":
    train_test()