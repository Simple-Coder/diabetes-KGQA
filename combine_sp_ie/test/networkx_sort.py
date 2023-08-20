"""
Created by xiedong
@Date: 2023/8/20 15:15
"""
# import networkx as nx
#
# # 创建一个有向图
# graph = nx.DiGraph()
#
# # 添加节点和建立关系（修正填反的关系）
# graph.add_edge("Drug_Disease", "Duration_Drug")
# graph.add_edge("Drug_Disease", "Method_Drug")
# graph.add_edge("Drug_Disease", "Frequency_Drug")
# graph.add_edge("Drug_Disease", "ADE_Drug")
# graph.add_edge("Drug_Disease", "Amount_Drug")
# graph.add_edge("Symptom_Disease", "Drug_Disease")
# graph.add_edge("Drug_Disease", "Reason_Disease")
# # 添加更多关联关系...
#
# # 获取拓扑排序结果
# topological_order = list(nx.topological_sort(graph))

# print("意图的拓扑排序顺序：", topological_order)
# print("nodes：", list(graph.nodes))
# print("edges：", list(graph.edges))
#
# successors = list(graph.successors('Drug_Disease'))
# print("successors of 'Drug_Disease':", successors)
#
# # 前驱节点
# # predecessors = list(graph.predecessors('Symptom_Disease'))
# # print("predecessors of 'Symptom_Disease':", predecessors)
# # 后继节点
# successors = list(graph.successors('Symptom_Disease'))
# print("successors of 'Symptom_Disease':", successors)


import networkx as nx


def find_topological_chains(intents, graph):
    chains = []

    def dfs(node, chain):
        nonlocal chains
        chain.append(node)

        if graph.out_degree(node) == 0:
            chains.append("->".join(chain))

        for successor in graph.successors(node):
            dfs(successor, chain.copy())

    for intent in intents:
        dfs(intent, [])

    return chains


# 创建一个有向图
graph = nx.DiGraph()
graph.add_edge("Drug_Disease", "Duration_Drug")
graph.add_edge("Drug_Disease", "Method_Drug")
graph.add_edge("Drug_Disease", "Frequency_Drug")
graph.add_edge("Drug_Disease", "ADE_Drug")
graph.add_edge("Drug_Disease", "Amount_Drug")
graph.add_edge("Symptom_Disease", "Drug_Disease")
graph.add_edge("Drug_Disease", "Reason_Disease")

intents = ["Drug_Disease", "Method_Drug", "Symptom_Disease"]

# 查找拓扑关系链并返回
chains = find_topological_chains(intents, graph)

filtered_chains = []
for chain in chains:
    chain_parts = chain.split("->")
    if set(intents) == set(chain_parts):
        filtered_chains.append(chain)

if filtered_chains:
    for chain in filtered_chains:
        print("拓扑关系链：", chain)
else:
    print("没有足够的关系链满足输入的意图。")
