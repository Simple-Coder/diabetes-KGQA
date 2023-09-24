import os
import sys
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        action_prob = self.policy_nn(state)
        return action_prob

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

    def compute_loss(self, action_prob, action):
        pass


def train_deep_path():
    torch.manual_seed(0)
    policy_network = SupervisedPolicy(state_dim, action_space)

    with open(relationPath, 'r') as f:
        train_data = f.readlines()

    num_samples = len(train_data)

    if num_samples > 500:
        num_samples = 500
    else:
        num_episodes = num_samples

    for episode in range(num_samples):
        print("Episode %d" % episode)
        print('Training Sample:', train_data[episode % num_samples][:-1])

        env = Env(dataPath, train_data[episode % num_samples])
        sample = train_data[episode % num_samples].split()

        try:
            good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
        except Exception as e:
            print('Cannot find a path')
            continue

        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)
                action_batch.append(transition.action)
            state_batch = np.array(state_batch, dtype=np.float32)  # 将列表的 NumPy 数组转换为单个 NumPy 数组
            state_batch = torch.tensor(state_batch)  # 转换为 PyTorch 张量
            state_batch = state_batch.squeeze(1)  # 调整输入数据的形状

            action_batch = torch.tensor(action_batch, dtype=torch.int64)

            # 更新策略
            loss = policy_network.update(state_batch, action_batch)

        # 保存模型
    torch.save(policy_network.state_dict(), 'models/policy_supervised_' + relation)
    print('Model saved')


def test(test_episodes):
    policy_network = SupervisedPolicy(state_dim, action_space)

    f = open(relationPath)
    test_data = f.readlines()
    f.close()

    test_num = len(test_data)

    test_data = test_data[-test_episodes:]
    print(len(test_data))

    success = 0

    # 加载模型参数
    model_path = os.path.join('models', 'policy_supervised_' + relation)
    policy_network.load_state_dict(torch.load(model_path))
    policy_network.eval()  # 设置模型为评估模式

    for episode, test_sample in enumerate(test_data):
        print(f'Test sample {episode}: {test_sample[:-1]}')
        env = Env(dataPath, test_sample)  # 请确保定义 Env 类
        sample = test_sample.split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        for t in count():
            state_vec = env.idx_state(state_idx)
            state_tensor = torch.FloatTensor(state_vec)

            with torch.no_grad():
                action_probs = policy_network(state_tensor)

            action_chosen = np.random.choice(np.arange(action_space), p=action_probs.numpy()[0])

            reward, new_state, done = env.interact(state_idx, action_chosen)

            if done or t == max_steps_test:
                if done:
                    print('Success')
                    success += 1
                print('Episode ends\n')
                break
            state_idx = new_state

    print(f'Success percentage: {success / test_episodes}')


if __name__ == '__main__':
    # 示例用法:
    # state_dim = 4  # 替换为实际状态维度
    # action_space = 2  # 替换为实际动作空间大小
    # policy = SupervisedPolicy(state_dim, action_space)
    #
    # # 更新策略
    # sample_state = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    # sample_action = torch.tensor([0], dtype=torch.int64)  # 假设只有一个动作
    # loss = policy.update(sample_state, sample_action)
    #
    # print(loss)

    train_deep_path()
    # test(500)
