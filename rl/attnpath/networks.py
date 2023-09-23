import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor


class policy_nn(nn.Module):
    def __init__(self, state_dim, action_dim, initializer='xavier', dropout1=0.0, dropout2=0.0, relation=None):
        super(policy_nn, self).__init__()
        self.linear1 = nn.Linear(state_dim, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        if initializer == 'xavier':
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.linear3.weight)

            self.linear1.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear1.bias.unsqueeze(0)).squeeze(0))
            self.linear2.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear2.bias.unsqueeze(0)).squeeze(0))
            self.linear3.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear3.bias.unsqueeze(0)).squeeze(0))

        self.relation = relation

    def forward(self, state):
        out = self.dropout1(state)
        out = self.linear1(state)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out  # Out is action_prob


class policy_nn_finetune(nn.Module):
    def __init__(self, state_dim, action_dim, entity_count, relation_count, dim=100, initializer='xavier',
                 dropout_rate=0.0):
        super(policy_nn_finetune, self).__init__()
        self.linear1 = nn.Linear(state_dim, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        if initializer == 'xavier':
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.linear3.weight)

            self.linear1.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear1.bias.unsqueeze(0)).squeeze(0))
            self.linear2.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear2.bias.unsqueeze(0)).squeeze(0))
            self.linear3.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear3.bias.unsqueeze(0)).squeeze(0))

        self.ent_embeddings = nn.Embedding(entity_count, dim)
        self.rel_embeddings = nn.Embedding(relation_count, dim)
        """
        ent_weight = np.loadtxt("../NELL-995/NELL-995_100_1.0_TransE_entity_embedding.txt")
        ent_weight = floatTensor(ent_weight)
        rel_weight = np.loadtxt("../NELL-995/NELL-995_100_1.0_TransE_relation_embedding.txt")
        rel_weight = floatTensor(rel_weight)
        """
        ent_weight = floatTensor(entity_count, dim)
        rel_weight = floatTensor(relation_count, dim)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

    def idx_state(self, idx_list):
        if idx_list != None:
            curr = self.ent_embeddings(longTensor([elem[0] for elem in idx_list]))
            targ = self.ent_embeddings(longTensor([elem[1] for elem in idx_list]))
            return torch.cat([curr, targ - curr], dim=1)

    def path_embedding(self, path):
        embeddings = self.ent_embeddings(longTensor(path))
        path_encoding = torch.sum(embeddings, dim=0)
        return torch.reshape(path_encoding, (-1, self.rel_embeddings.weight.shape[1]))

    def forward(self, idx_list):
        state = self.idx_state(idx_list)
        out = self.linear1(state)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out  # Out is action_prob


class policy_nn_dropout(nn.Module):
    def __init__(self, state_dim, action_dim, dropout1=0.2, dropout2=0.5, initializer='xavier'):
        super(policy_nn_dropout, self).__init__()
        self.linear1 = nn.Linear(state_dim, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        if initializer == 'xavier':
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.linear3.weight)

            self.linear1.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear1.bias.unsqueeze(0)).squeeze(0))
            self.linear2.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear2.bias.unsqueeze(0)).squeeze(0))
            self.linear3.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear3.bias.unsqueeze(0)).squeeze(0))

    def forward(self, state):
        out = self.dropout1(state)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out  # Out is action_prob


class value_nn(nn.Module):
    def __init__(self, state_dim, initializer='xavier'):
        super(value_nn, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        if initializer == 'xavier':
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

            self.linear1.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear1.bias.unsqueeze(0)).squeeze(0))
            self.linear2.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear2.bias.unsqueeze(0)).squeeze(0))

    def forward(self, state):
        out = self.linear1(state)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class q_network(nn.Module):
    def __init__(self, state_dim, action_space, initializer='xavier'):
        super(q_network, self).__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, action_space)
        self.relu = nn.ReLU()

        if initializer == 'xavier':
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.linear3.weight)

            self.linear1.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear1.bias.unsqueeze(0)).squeeze(0))
            self.linear2.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear2.bias.unsqueeze(0)).squeeze(0))
            self.linear3.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear3.bias.unsqueeze(0)).squeeze(0))

    def forward(self, state):
        out = self.linear1(state)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out


class policy_nn_lstm(nn.Module):  # Modified to embedding_dim so as to easily control the multiples
    def __init__(self, embedding_dim, action_dim, initializer='xavier', dropout_rate=0.0, multiplier=4, relation=None):
        super(policy_nn_lstm, self).__init__()
        self.linear1 = nn.Linear(embedding_dim * multiplier, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.start_relation = nn.Parameter(torch.randn(1, embedding_dim))
        self.lstm = nn.LSTM(input_size=2 * embedding_dim, hidden_size=2 * embedding_dim, num_layers=3)
        self.dropout = nn.Dropout(dropout_rate)

        if initializer == 'xavier':
            for parameter in self.linear1.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            for parameter in self.linear2.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            for parameter in self.linear3.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            for parameter in self.lstm.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            nn.init.xavier_uniform_(self.start_relation)

        self.relation = relation

    def forward(self, state, lstm_input, hidden, cell):
        """
        state: (seq_len, input_size + hidden_size)
        lstm_input: (seq_len, batch, input_size)
        hidden: (num_layers * num_directions, batch, hidden_size)
        cell: (num_layers * num_directions, batch, hidden_size)
        """

        lstm_output, (hidden_new, cell_new) = self.lstm(lstm_input, (hidden, cell))
        """
        lstm_output: (seq_len, batch, num_directions * hidden_size)
        hidden_new: (num_layers * num_directions, batch, hidden_size)
        cell_new: (num_layers * num_directions, batch, hidden_size)
        """

        out = self.linear1(state)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out, lstm_output, hidden_new, cell_new  # Out is action_prob


class policy_nn_attn(nn.Module):
    def __init__(self, embedding_dim, action_dim, attn_dim, initializer='xavier', dropout1=0.0, dropout2=0.0,
                 leakyrelu_slope=0.2, relation=None):
        super(policy_nn_attn, self).__init__()
        self.linear1 = nn.Linear(embedding_dim * 3, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        self.attn = nn.Linear(embedding_dim, attn_dim)
        self.attn_weight = nn.Linear(attn_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(leakyrelu_slope)

        if initializer == 'xavier':
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.linear3.weight)

            self.linear1.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear1.bias.unsqueeze(0)).squeeze(0))
            self.linear2.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear2.bias.unsqueeze(0)).squeeze(0))
            self.linear3.bias = nn.Parameter(nn.init.xavier_uniform_(self.linear3.bias.unsqueeze(0)).squeeze(0))

            nn.init.xavier_uniform_(self.attn.weight)
            nn.init.xavier_uniform_(self.attn_weight.weight)

        self.relation = relation
        self.attn_dim = attn_dim

    def forward(self, state, now_embedding, neighbour_embeddings_list):
        neighbour_cnt_list = [neighbour_embeddings.shape[0] for neighbour_embeddings in neighbour_embeddings_list]
        split_point_list = [0] + np.cumsum(neighbour_cnt_list).tolist()

        now_projection = self.attn(now_embedding)
        neighbour_projection = self.attn(torch.cat(neighbour_embeddings_list, dim=0))
        now_projection = [projection.expand(neighbour_cnt, self.attn_dim) for (projection, neighbour_cnt) in
                          zip(now_projection, neighbour_cnt_list)]
        now_projection = torch.cat(now_projection, dim=0)
        concat_projection = torch.cat([now_projection, neighbour_projection], dim=1)
        concat_projection = self.attn_weight(concat_projection)
        concat_projection = self.leakyrelu(concat_projection)
        concat_projection = concat_projection.squeeze(1)
        concat_projection_list = [concat_projection[split_point_list[i]: split_point_list[i + 1]] for i in
                                  range(len(neighbour_cnt_list))]
        concat_projection_list = [self.softmax(concat_projection).unsqueeze(0) for concat_projection in
                                  concat_projection_list]  # Attention weight of each neighbour node
        concat_projection_list = [torch.mm(concat_projection, neighbour_embeddings) for
                                  (concat_projection, neighbour_embeddings) in zip(concat_projection_list,
                                                                                   neighbour_embeddings_list)]  # Hidden vector representation of the current node
        concat_projection = torch.cat(concat_projection_list, dim=0)

        out = torch.cat([state, concat_projection], dim=1)
        out = self.dropout1(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out  # Out is action_prob


class policy_nn_lstm_attn(nn.Module):
    def __init__(self, embedding_dim, action_dim, attn_dim, initializer='xavier', dropout_rate=0.0, leakyrelu_slope=0.2,
                 multiplier=5, relation=None):
        super(policy_nn_lstm_attn, self).__init__()
        self.linear1 = nn.Linear(embedding_dim * multiplier, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.start_relation = nn.Parameter(torch.randn(1, embedding_dim))
        self.lstm = nn.LSTM(input_size=2 * embedding_dim, hidden_size=2 * embedding_dim, num_layers=3)
        self.dropout = nn.Dropout(dropout_rate)

        self.attn = nn.Linear(embedding_dim, attn_dim)
        self.attn_weight = nn.Linear(attn_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(leakyrelu_slope)

        if initializer == 'xavier':
            for parameter in self.linear1.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            for parameter in self.linear2.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            for parameter in self.linear3.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            for parameter in self.lstm.parameters():
                if len(parameter.shape) == 1:
                    parameter = nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))
                else:
                    nn.init.xavier_uniform_(parameter)

            nn.init.xavier_uniform_(self.start_relation)

            nn.init.xavier_uniform_(self.attn.weight)
            nn.init.xavier_uniform_(self.attn_weight.weight)

        self.relation = relation
        self.attn_dim = attn_dim

    def forward(self, state, lstm_input, hidden, cell, now_embedding, neighbour_embeddings_list):
        """
        state: (seq_len, input_size + hidden_size)
        lstm_input: (seq_len, batch, input_size)
        hidden: (num_layers * num_directions, batch, hidden_size)
        cell: (num_layers * num_directions, batch, hidden_size)
        """

        lstm_output, (hidden_new, cell_new) = self.lstm(lstm_input, (hidden, cell))
        """
        lstm_output: (seq_len, batch, num_directions * hidden_size)
        hidden_new: (num_layers * num_directions, batch, hidden_size)
        cell_new: (num_layers * num_directions, batch, hidden_size)
        """

        neighbour_cnt_list = [neighbour_embeddings.shape[0] for neighbour_embeddings in neighbour_embeddings_list]
        split_point_list = [0] + np.cumsum(neighbour_cnt_list).tolist()

        now_projection = self.attn(now_embedding)
        neighbour_projection = self.attn(torch.cat(neighbour_embeddings_list, dim=0))
        now_projection = [projection.expand(neighbour_cnt, self.attn_dim) for (projection, neighbour_cnt) in
                          zip(now_projection, neighbour_cnt_list)]
        now_projection = torch.cat(now_projection, dim=0)
        concat_projection = torch.cat([now_projection, neighbour_projection], dim=1)
        concat_projection = self.attn_weight(concat_projection)
        concat_projection = self.leakyrelu(concat_projection)
        concat_projection = concat_projection.squeeze(1)
        concat_projection_list = [concat_projection[split_point_list[i]: split_point_list[i + 1]] for i in
                                  range(len(neighbour_cnt_list))]
        concat_projection_list = [self.softmax(concat_projection).unsqueeze(0) for concat_projection in
                                  concat_projection_list]  # Attention weight of each neighbour node
        concat_projection_list = [torch.mm(concat_projection, neighbour_embeddings) for
                                  (concat_projection, neighbour_embeddings) in zip(concat_projection_list,
                                                                                   neighbour_embeddings_list)]  # Hidden vector representation of the current node
        concat_projection = torch.cat(concat_projection_list, dim=0)

        out = torch.cat([state, concat_projection], dim=1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out, lstm_output, hidden_new, cell_new  # Out is action_prob
