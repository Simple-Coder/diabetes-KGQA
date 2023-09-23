import os
import time

import numpy as np
import random
from utils import *

import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor
    byteTensor = torch.cuda.ByteTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor
    byteTensor = torch.ByteTensor


class Env(object):
    """knowledge graph environment definition"""

    def __init__(self, dataPath, task=None, model="TransE"):
        f1 = open(dataPath + 'entity2id.txt')
        f2 = open(dataPath + 'relation2id.txt')
        self.entity2id = f1.readlines()
        self.relation2id = f2.readlines()
        f1.close()
        f2.close()
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.id2entity_ = {}
        self.id2relation_ = {}
        self.relations = []

        for line in self.entity2id:
            self.entity2id_[line.split()[0]] = int(line.split()[1])
            self.id2entity_[int(line.split()[1])] = line.split()[0]

        for line in self.relation2id:
            self.relation2id_[line.split()[0]] = int(line.split()[1])
            self.id2relation_[int(line.split()[1])] = line.split()[0]
            self.relations.append(line.split()[0])

        # Which model to compute pretrained embedding of entities and relations? (The definition of states)
        if model == "TransH":
            print("Uses TransH")
            self.entity2vec = np.loadtxt(dataPath + 'NELL-995_100_1.0_TransH_entity_embedding.txt')
            self.relation2vec = np.loadtxt(dataPath + 'NELL-995_100_1.0_TransH_relation_embedding.txt')
            self.norm2vec = np.loadtxt(dataPath + 'NELL-995_100_1.0_TransH_norm_embedding.txt')
            if task is not None:
                relation = task.strip().split()[2].replace('_', ':')
                w_r = self.norm2vec[self.relation2id_[relation]]
                new_entity2vec = self.entity2vec - np.sum(self.entity2vec * w_r, axis=1, keepdims=True) * w_r
                self.entity2vec = new_entity2vec

        elif model == "TransR":
            print("Uses TransR")
            self.entity2vec = np.loadtxt(dataPath + 'NELL-995_100_1.0_TransR_entity_embedding.txt')
            self.relation2vec = np.loadtxt(dataPath + 'NELL-995_100_1.0_TransR_relation_embedding.txt')
            self.projection2vec = np.loadtxt(dataPath + "NELL-995_100_1.0_TransR_norm_embedding.txt")
            dim = int(np.sqrt(self.projection2vec.shape[1]))
            # By default, entities and relations share the same dimension
            # This is not the main point of research
            self.projection2vec = self.projection2vec.reshape([-1, dim, dim])
            if task is not None:
                relation = task.strip().split()[2].replace('_', ':')
                M_vec = self.projection2vec[self.relation2id_[relation], :, :]
                new_entity2vec = np.matmul(M_vec, self.entity2vec.T).T
                self.entity2vec = new_entity2vec

        elif model == "TransD":
            print("Uses TransD")
            self.entity2vec = np.loadtxt(dataPath + 'NELL-995_100_1.0_TransD_entity_embedding.txt')
            self.relation2vec = np.loadtxt(dataPath + 'NELL-995_100_1.0_TransD_relation_embedding.txt')
            self.ent_norm2vec = np.loadtxt(dataPath + "NELL-995_100_1.0_TransD_ent_norm_embedding.txt")
            self.rel_norm2vec = np.loadtxt(dataPath + "NELL-995_100_1.0_TransD_rel_norm_embedding.txt")
            if task is not None:
                relation = task.strip().split()[2].replace('_', ':')
                rel_proj = self.rel_norm2vec[self.relation2id_[relation]]
                new_entity2vec = self.entity2vec + np.sum(self.entity2vec * self.ent_norm2vec, axis=1,
                                                          keepdims=True) * rel_proj
                self.entity2vec = new_entity2vec

        elif model == "ProjE":
            print("Uses ProjE")
            self.entity2vec = np.loadtxt(dataPath + 'NELL-995_100_ProjE_entity_embedding.txt')
            self.relation2vec = np.loadtxt(dataPath + 'NELL-995_100_ProjE_relation_embedding.txt')
            self.simple_hr_combination_weights = np.loadtxt(
                dataPath + "NELL-995_100_ProjE_simple_hr_combination_weights.txt")
            self.simple_tr_combination_weights = np.loadtxt(
                dataPath + "NELL-995_100_ProjE_simple_tr_combination_weights.txt")
            self.combination_bias_hr = np.loadtxt(dataPath + "NELL-995_100_ProjE_combination_bias_hr.txt")
            self.combination_bias_tr = np.loadtxt(dataPath + "NELL-995_100_ProjE_combination_bias_tr.txt")
            if task is not None:
                relation = task.strip().split()[2].replace('_', ':')
                dim = self.entity2vec.shape[1]
                r = self.relation2vec[[self.relation2id_[relation]]]
                # ent_mat = np.transpose(self.entity2vec)
                hr = self.entity2vec * self.simple_hr_combination_weights[
                                       :dim] + r * self.simple_hr_combination_weights[dim:]
                new_entity2vec = np.tanh(hr + self.combination_bias_hr)
                self.entity2vec = new_entity2vec

        # elif model == "ConvE":
        #     print("Uses ConvE")
        #     start_time = time.time()
        #     self.entity2vec = np.loadtxt(dataPath + 'NELL-995_100_ConvE_entity_embedding.txt')
        #     self.relation2vec = np.loadtxt(dataPath + 'NELL-995_100_ConvE_relation_embedding.txt')
        #
        #     self.TransE_to_ConvE_id_entity = {}
        #     with open(dataPath + "TransE_to_ConvE_entity_id.txt") as fr:
        #         for line in fr:
        #             line_list = line.strip().split()
        #             self.TransE_to_ConvE_id_entity[int(line_list[0])] = int(line_list[1])
        #
        #     self.TransE_to_ConvE_id_relation = {}
        #     with open(dataPath + "TransE_to_ConvE_relation_id.txt") as fr:
        #         for line in fr:
        #             line_list = line.strip().split()
        #             self.TransE_to_ConvE_id_relation[int(line_list[0])] = int(line_list[1])
        #
        #     homepath = os.path.expanduser('~')
        #     token2idx_ent, idx2token_ent, label2idx_ent, idx2label_ent = pickle.load(
        #         open(homepath + "/.data/NELL-995/vocab_e1", 'rb'))
        #     token2idx_rel, idx2token_rel, label2idx_rel, idx2label_rel = pickle.load(
        #         open(homepath + "/.data/NELL-995/vocab_rel", 'rb'))
        #
        #     self.ConvE_model = ConvE_double(len(token2idx_ent), len(token2idx_rel))
        #     model_params = torch.load(dataPath + "NELL-995_ConvE_0.2_0.3_100.model")
        #     self.ConvE_model.load_state_dict(model_params)
        #
        #     for parameter in self.ConvE_model.parameters():
        #         parameter.requires_grad = False
        #
        #     if USE_CUDA:
        #         self.ConvE_model.cuda()
        #
        #     if task is not None:
        #         relation = task.strip().split()[2].replace('_', ':')
        #
        #         rel_id = token2idx_rel[relation]
        #
        #         ConvE_ent_id_list = [self.TransE_to_ConvE_id_entity[i] for i in
        #                              range(len(self.TransE_to_ConvE_id_entity))]
        #         new_entity2vec_list = []
        #         bs = self.ConvE_model.batch_size
        #         batch_count = len(ConvE_ent_id_list) // bs
        #         for i in range(batch_count):
        #             x_middle, output = self.ConvE_model(longTensor(ConvE_ent_id_list[i * bs: (i + 1) * bs]),
        #                                                 longTensor([rel_id] * bs))
        #             new_entity2vec_list.append(x_middle.cpu())
        #         if len(ConvE_ent_id_list) % bs != 0:
        #             input_ent_list = ConvE_ent_id_list[batch_count * bs:] + [0] * (bs - len(ConvE_ent_id_list) % bs)
        #             x_middle, output = self.ConvE_model(longTensor(input_ent_list), longTensor([rel_id] * bs))
        #             new_entity2vec_list.append(x_middle[: len(ConvE_ent_id_list) % bs].cpu())
        #         self.entity2vec = torch.cat(new_entity2vec_list).numpy()
        #
        #         torch.cuda.empty_cache()
        #
        #     """
        #     else:
        #         if USE_CUDA:
        #             self.ConvE_model.cuda()
        #     """
        #
        #     end_time = time.time()
        #     print("Embedding calculation time: ", end_time - start_time)
        else:
            print("Default. Uses TransE")
            self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')
            self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')

        if task is None:
            self.embedding_precomputed_flag = False
        else:
            self.embedding_precomputed_flag = True

        self.model = model

        self.path = []
        self.path_relations = []

        # Knowledge Graph for path finding
        f = open(dataPath + 'kb_env_rl.txt')
        kb_all = f.readlines()
        f.close()

        self.kb = []
        if task != None:
            relation = task.split()[2]  # Remove query relation and its inverse
            for line in kb_all:
                rel = line.split()[2]
                if rel != relation and rel != relation + '_inv':
                    self.kb.append(line)
        else:
            for line in kb_all:
                self.kb.append(line)

        self.entity2link = {}

        # Build the dictionary. Attention: they are all represented with numbers!
        for line in self.kb:
            line_list = line.strip().split()
            head = self.entity2id_[line_list[0]]
            tail = self.entity2id_[line_list[1]]
            rel = self.relation2id_[line_list[2]]

            if head not in self.entity2link:
                self.entity2link[head] = {rel: [tail]}
            elif rel not in self.entity2link[head]:
                self.entity2link[head][rel] = [tail]
            else:
                self.entity2link[head][rel].append(tail)

        self.die = 0  # record how many times does the agent choose an invalid action

        self.banned_action_list = []

    def interact(self, state, action):
        # state and action are all represented with numbers

        # print("Die: ", self.die)
        '''
        This function process the interact from the agent
        state: is [current_position, target_position, die]
        action: an integer
        return: (reward, [new_position, target_position, die], done)
        '''
        done = 0  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]

        if action in self.banned_action_list:
            # print("Type 1")
            choices = []
        elif curr_pos not in self.entity2link:
            # print("Type 2", curr_pos)
            choices = []
        elif action not in self.entity2link[curr_pos]:
            # print("Type 3")
            choices = []
        else:
            # print("Type 4")
            choices = self.entity2link[curr_pos][action]

        """
        chosed_relation = self.relations[action]
        choices = []
        for line in self.kb:
            triple = line.rsplit()
            e1_idx = self.entity2id_[triple[0]] 
            
            if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id_:
                choices.append(triple) 
        """

        if len(choices) == 0:  # doesn't find a successful path
            # print("No proper path! ")
            reward = -1
            self.die += 1
            next_state = state  # stay in the initial state
            next_state[-1] = self.die  # Total failure times
            # print(next_state)
            return (reward, next_state, done)
        else:  # find a valid step
            # print("Proper path exists! ")
            chose_entity = random.choice(choices)  # Randomly choose one from multiple choices
            self.path.append(self.id2relation_[action] + ' -> ' + self.id2entity_[
                chose_entity])  # path[2]: relation；path[1]: tail entity（the next entity）
            self.path_relations.append(self.id2relation_[action])  # Relation
            # print 'Find a valid step', path
            # print 'Action index', action
            self.die = 0
            new_pos = chose_entity  # Using the next entity as the new position
            reward = 0  # Reward is zero means the action is valid
            new_state = [new_pos, target_pos, self.die]

            if new_pos == target_pos:
                print('Find a path:', self.path)
                done = 1  # episode finished
                reward = 0  # reward is 0 means the episode is successful
                new_state = None
            # print(new_state)
            return (reward, new_state, done)

    def idx_state(self, idx_list, relation=None):  # Calculate state vector
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]
            targ = self.entity2vec[idx_list[1], :]

            if self.embedding_precomputed_flag == True or relation is None:
                pass
            else:
                if self.model == "TransH":
                    w_r = self.norm2vec[relation]
                    curr = curr - np.sum(curr * w_r) * w_r
                    targ = targ - np.sum(targ * w_r) * w_r
                elif self.model == "TransR":
                    M_vec = self.projection2vec[relation, :, :]
                    curr = np.matmul(M_vec, curr.T).T
                    targ = np.matmul(M_vec, targ.T).T
                elif self.model == "TransD":
                    rel_proj = self.rel_norm2vec[relation]
                    curr = curr + np.sum(curr * self.ent_norm2vec[idx_list[0]]) * rel_proj
                    targ = targ + np.sum(targ * self.ent_norm2vec[idx_list[1]]) * rel_proj
                elif self.model == "ProjE":
                    dim = self.entity2vec.shape[1]
                    r = self.relation2vec[relation]
                    curr = curr * self.simple_hr_combination_weights[:dim] + r * self.simple_hr_combination_weights[
                                                                                 dim:]
                    curr = np.tanh(curr + self.combination_bias_hr)
                    targ = targ * self.simple_hr_combination_weights[:dim] + r * self.simple_hr_combination_weights[
                                                                                 dim:]
                    targ = np.tanh(targ + self.combination_bias_hr)
                elif self.model == "ConvE":
                    curr_id = self.TransE_to_ConvE_id_entity[idx_list[0]]
                    targ_id = self.TransE_to_ConvE_id_entity[idx_list[1]]
                    rel_id = self.TransE_to_ConvE_id_relation[relation]
                    bs = self.ConvE_model.batch_size
                    curr = [curr_id] + [0] * (bs - 1)
                    curr, output = self.ConvE_model(longTensor(curr), longTensor([rel_id] * bs))
                    curr = curr[0].cpu().numpy()
                    targ = [targ_id] + [0] * (bs - 1)
                    targ, output = self.ConvE_model(longTensor(targ), longTensor([rel_id] * bs))
                    targ = targ[0].cpu().numpy()
                else:  # Default, TransE
                    pass

            return np.expand_dims(np.concatenate((curr, targ - curr)), axis=0)
        else:
            return None

    def get_valid_actions(self, entityID):  # Get the valid action
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path):  # A path's embedding is calculated as summing all the relational vectors
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        embeddings = np.reshape(embeddings, (-1, embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, embedding_dim))
