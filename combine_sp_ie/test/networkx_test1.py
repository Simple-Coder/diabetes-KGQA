"""
Created by xiedong
@Date: 2023/8/20 16:15
"""
from itertools import combinations, permutations

# 定义节点列表
nodes = ["Drug_Disease", "Method_Drug", "Pathogenesis_Disease"]

for r in range(1, len(nodes) + 1):
    permutations_set = set(permutations(nodes, r))
    # 输出所有可能的无序排列
    for perm in permutations_set:
        perm_str = '->'.join(perm)
        print(perm_str)
    print("层级结束:", r)
