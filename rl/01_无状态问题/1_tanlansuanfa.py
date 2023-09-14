"""
Created by xiedong
@Date: 2023/9/14 16:50
"""
import numpy as np

# 每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)

# 记录每个老虎机的返回值
rewards = [[1] for _ in range(10)]

import random


# 贪婪算法
def choose_one():
    # 有小概率随机选择一根拉杆
    if random.random() < 0.01:
        return random.randint(0, 9)

    # 计算每个老虎机的奖励平均
    rewards_mean = [np.mean(i) for i in rewards]

    # 选择期望奖励估值最大的拉杆
    return np.argmax(rewards_mean)


one = choose_one()


def try_and_play():
    i = choose_one()

    # 玩老虎机,得到结果
    reward = 0
    if random.random() < probs[i]:
        reward = 1

    # 记录玩的结果
    rewards[i].append(reward)


try_and_play()


def get_result():
    # 玩N次
    for _ in range(5000):
        try_and_play()

    # 期望的最好结果
    target = probs.max() * 5000

    # 实际玩出的结果
    result = sum([sum(i) for i in rewards])

    return target, result


result = get_result()
print()
