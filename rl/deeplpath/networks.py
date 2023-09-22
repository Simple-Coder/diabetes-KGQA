import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, action_dim)

    def forward(self, state):
        # 第一层全连接层
        h1 = F.relu(self.fc1(state))
        # 第二层全连接层
        h2 = F.relu(self.fc2(h1))
        # 输出层，使用softmax
        action_prob = F.softmax(self.fc3(h2), dim=-1)
        return action_prob


class ValueNN(nn.Module):
    def __init__(self, state_dim):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state):
        # 第一层全连接层
        h1 = F.relu(self.fc1(state))
        # 输出层
        value_estimated = self.fc2(h1)
        return value_estimated.squeeze()


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, state):
        # 第一层全连接层
        h1 = F.relu(self.fc1(state))
        # 第二层全连接层
        h2 = F.relu(self.fc2(h1))
        # 输出层
        action_values = self.fc3(h2)
        return action_values


# # 创建PolicyNN、ValueNN和QNetwork的实例
# state_dim = 200  # 请填入state_dim的值
# action_dim = 400  # 请填入action_dim的值
# action_space = 200  # 请填入action_space的值
#
# policy_nn = PolicyNN(state_dim, action_dim)
# value_nn = ValueNN(state_dim)
# q_network = QNetwork(state_dim, action_space)
#
# # 创建一个虚拟输入张量（示例）
# dummy_state = torch.randn(1, state_dim)  # 使用随机数据创建一个虚拟状态张量
#
# # 前向传播示例
# policy_output = policy_nn(dummy_state)
# value_output = value_nn(dummy_state)
# q_values = q_network(dummy_state)
#
# # 打印模型输出
# print("Policy Output:")
# print(policy_output)
#
# print("Value Output:")
# print(value_output)
#
# print("Q Values:")
# print(q_values)
