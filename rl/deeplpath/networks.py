"""
Created by xiedong
@Date: 2023/9/19 15:46
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNN(nn.Module):

    def __init__(self, state_dim, action_dim, initializer=None):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512, bias=True)
        self.fc2 = nn.Linear(512, 1024, bias=True)
        self.fc3 = nn.Linear(1024, action_dim, bias=True)
        self.softmax = nn.Softmax(action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs


class ValueNN(nn.Module):

    def __init__(self, state_dim, initializer=None):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64, bias=True)
        self.fc2 = nn.Linear(64, 1, bias=True)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        value_estimated = self.fc2(x)
        return torch.squeeze(value_estimated)


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_values = self.fc3(x)
        return action_values
