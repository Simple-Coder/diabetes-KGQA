"""
Created by xiedong
@Date: 2023/8/20 15:00
"""
import networkx as nx

# 创建一个有向图
graph = nx.DiGraph()

# 添加所有意图节点
intents = [
    "Method_Drug", "Duration_Drug", "Frequency_Drug", "ADE_Drug", "Amount_Drug",
    "Drug_Disease", "Symptom_Disease", "Reason_Disease", "Anatomy_Disease",
    "Class_Disease", "Operation_Disease", "Pathogenesis_Disease", "Test_Disease",
    "Test_items_Disease", "Treatment_Disease"
]
graph.add_nodes_from(intents)

# 建立意图之间的关联关系
graph.add_edge("Drug_Disease", "Method_Drug", relation="用药方法->药品名称")
graph.add_edge("Drug_Disease", "Duration_Drug", relation="持续时间->药品名称")
graph.add_edge("Drug_Disease", "Frequency_Drug", relation="用药频率->药品名称")
graph.add_edge("Drug_Disease", "ADE_Drug", relation="不良反应->药品名称")
graph.add_edge("Drug_Disease", "Amount_Drug", relation="用药剂量->药品名称")


def find_related_intents(intent_name, graph):
    if not graph.has_node(intent_name):
        return intent_name
    related_intents = []
    for neighbor in graph.successors(intent_name):
        relation = graph.get_edge_data(intent_name, neighbor).get("relation")
        related_intents.append((neighbor, relation))
    return related_intents


# 查询临床表现相关的意图
symptom_intent = "Symptom_Disease"
related_intents = find_related_intents(symptom_intent, graph)
print(f"与{symptom_intent}相关的意图：", related_intents)

# 查询药品名称相关的意图
drug_intent = "Drug_Disease"
related_intents = find_related_intents(drug_intent, graph)
print(f"与{drug_intent}相关的意图：", related_intents)
