import numpy as np
import random
from utils import *


class Env(object):
    """知识图谱环境定义"""

    def __init__(self, dataPath, task=None):
        f1 = open(dataPath + 'entity2id.txt')
        f2 = open(dataPath + 'relation2id.txt')
        self.entity2id = f1.readlines()  # 读取实体到ID的映射
        self.relation2id = f2.readlines()  # 读取关系到ID的映射
        f1.close()
        f2.close()
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.relations = []
        for line in self.entity2id:
            self.entity2id_[line.split()[0]] = int(line.split()[1])
        for line in self.relation2id:
            self.relation2id_[line.split()[0]] = int(line.split()[1])
            self.relations.append(line.split()[0])
        self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')  # 读取实体向量
        self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')  # 读取关系向量

        self.path = []
        self.path_relations = []

        # 用于路径查找的知识图谱
        f = open(dataPath + 'kb_env_rl.txt')
        kb_all = f.readlines()
        f.close()

        self.kb = []
        if task != None:
            relation = task.split()[2]
            for line in kb_all:
                rel = line.split()[2]
                if rel != relation and rel != relation + '_inv':
                    self.kb.append(line) #筛选出知识图谱中与该关系相关的三元组，以便在环境中只考虑与任务相关的知识。这有助于减少环境的复杂性，提高代理在特定任务上的效率。

        self.die = 0  # 记录代理选择无效路径的次数

    def interact(self, state, action):
        '''
        该函数处理来自代理的交互
        state: [当前位置, 目标位置]
        action: 一个整数
        返回: (奖励, [新的位置, 目标位置], 是否完成)
        '''
        done = 0  # 本集是否完成
        curr_pos = state[0]
        target_pos = state[1]
        chosed_relation = self.relations[action]
        choices = []
        for line in self.kb:
            triple = line.rsplit()
            e1_idx = self.entity2id_[triple[0]]

            if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id_:
                choices.append(triple)
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state  # 保持在初始状态
            next_state[-1] = self.die
            return (reward, next_state, done)
        else:  # 找到有效的步骤
            path = random.choice(choices)
            self.path.append(path[2] + ' -> ' + path[1])
            self.path_relations.append(path[2])
            self.die = 0
            new_pos = self.entity2id_[path[1]]
            reward = 0
            new_state = [new_pos, target_pos, self.die]

            if new_pos == target_pos:
                print('找到路径:', self.path)
                done = 1
                reward = 0
                new_state = None
            return (reward, new_state, done)

    def idx_state(self, idx_list):
        """
        将状态索引转换为状态表示（嵌入）。

        参数：
        idx_list (list): 包含当前位置索引和目标位置索引的列表。

        返回：
        np.ndarray: 表示状态的嵌入向量。
        """
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]
            targ = self.entity2vec[idx_list[1], :]
            return np.expand_dims(np.concatenate((curr, targ - curr)), axis=0)
        else:
            return None

    def get_valid_actions(self, entityID):
        """
        获取特定实体的有效操作集合。

        参数：
        entityID (int): 实体的索引。

        返回：
        np.ndarray: 包含有效操作的整数数组。
        """
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path):
        """
        计算路径的嵌入表示。

        参数：
        path (list): 包含路径上关系的列表。

        返回：
        np.ndarray: 表示路径嵌入的向量。
        """
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        embeddings = np.reshape(embeddings, (-1, embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, embedding_dim))


if __name__ == "__main__":
    # 创建知识图谱环境
    dataPath = 'your_data_path/'  # 替换为你的数据路径
    task = 'task_description'  # 替换为你的任务描述
    env = Env(dataPath, task)

    # 定义初始状态
    initial_state = [0, 5, 0]  # [当前位置, 目标位置, 无效步数]

    done = False
    while not done:
        # 获取有效操作
        valid_actions = env.get_valid_actions(initial_state[0])

        if len(valid_actions) == 0:
            print("没有有效操作可选。")
            break

        # 随机选择一个有效操作
        action = np.random.choice(valid_actions)

        # 与环境交互
        reward, new_state, done = env.interact(initial_state, action)

        if done:
            print("任务完成！")
            break

        # 更新当前状态
        initial_state = new_state

    # 输出路径信息
    if env.path:
        print("路径信息：")
        for step, relation in enumerate(env.path):
            print(f"Step {step + 1}: {relation}")
