"""
Created by xiedong
@Date: 2023/8/20 15:15
"""
import networkx as nx

# 创建一个有向图
graph = nx.DiGraph()

# 添加节点和建立关系（修正填反的关系）
graph.add_edge("Drug_Disease", "Duration_Drug")
graph.add_edge("Drug_Disease", "Method_Drug")
graph.add_edge("Drug_Disease", "Frequency_Drug")
graph.add_edge("Drug_Disease", "ADE_Drug")
graph.add_edge("Drug_Disease", "Amount_Drug")
graph.add_edge("Symptom_Disease", "Drug_Disease")
graph.add_edge("Drug_Disease", "Reason_Disease")
# 添加更多关联关系...

# 获取拓扑排序结果
topological_order = list(nx.topological_sort(graph))

print("意图的拓扑排序顺序：", topological_order)
print("nodes：", list(graph.nodes))
print("edges：", list(graph.edges))

successors = list(graph.successors('Drug_Disease'))
print("successors of 'Drug_Disease':", successors)

# 前驱节点
# predecessors = list(graph.predecessors('Symptom_Disease'))
# print("predecessors of 'Symptom_Disease':", predecessors)
# 后继节点
successors = list(graph.successors('Symptom_Disease'))
print("successors of 'Symptom_Disease':", successors)
