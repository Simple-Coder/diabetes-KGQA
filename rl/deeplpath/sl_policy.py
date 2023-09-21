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


def train():
    torch.manual_seed(seed=0)  # 设置随机种子以确保结果的可重复性
    policy_nn = SupervisedPolicy(state_dim, action_space)

    with open(relationPath, 'r') as f:
        train_data = f.readlines()

    num_samples = len(train_data)

    if num_samples > 500:
        num_samples = 500
    else:
        num_samples = num_samples

    for episode in range(num_samples):
        print("Episode %d" % episode)
        print('Training Sample:', train_data[episode % num_samples][:-1])

        env = Env(dataPath, train_data[episode % num_samples])
        sample = train_data[episode % num_samples].split()

        try:
            good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
        except Exception as e:
            print('无法找到路径')
            continue

        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)
                action_batch.append(transition.action)
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, state_dim])

            loss = policy_nn.update(torch.Tensor(state_batch), torch.LongTensor(action_batch))
            print('Loss:', loss)

    torch.save(policy_nn.state_dict(), 'models/policy_supervised_' + relation + '.pt')
    print('模型已保存')


def test(test_episodes):
    torch.manual_seed(0)  # 设置随机种子以确保结果的可重复性
    policy_nn = SupervisedPolicy(state_dim, action_space)

    with open(relationPath, 'r') as f:
        test_data = f.readlines()

    test_num = len(test_data)

    test_data = test_data[-test_episodes:]
    print(len(test_data))

    success = 0

    # 加载已经训练好的模型
    policy_nn.load_state_dict(torch.load('models/policy_supervised_' + relation + '.pt'))
    policy_nn.eval()  # 将模型设置为评估模式
    print('模型已加载')

    for episode in range(len(test_data)):
        print('测试样本 %d: %s' % (episode, test_data[episode][:-1]))
        env = Env(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = policy_nn.predict(torch.Tensor(state_vec))
            action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs.numpy()))
            reward, new_state, done = env.interact(state_idx, action_chosen)

            if done or t == max_steps_test:
                if done:
                    print('成功')
                    success += 1
                print('本轮结束\n')
                break

            state_idx = new_state

    print('成功百分比:', success / test_episodes)


if __name__ == "__main__":
    train()
    # test(50)
