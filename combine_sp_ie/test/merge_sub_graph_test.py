"""
Created by xiedong
@Date: 2023/8/10 12:43
"""


def group_subgraphs_by_relation(subgraphs):
    grouped_subgraphs = {}

    for graph in subgraphs:
        relation_info = graph["info"]["relationship"]
        related_node = graph["info"]["related_node"]
        node = graph["info"]["node"]

        key = (relation_info, related_node)  # 使用关系和关联节点作为组合的键
        if key not in grouped_subgraphs:
            grouped_subgraphs[key] = {"relation": relation_info, "related_node": related_node, "subgraphs": []}

        grouped_subgraphs[key]["subgraphs"].append(node)

    grouped_results = [[group["related_node"], group["relation"], group["subgraphs"]] for group in grouped_subgraphs.values()]

    return grouped_results

# 假设 subgraphs 是你的子图列表
subgraphs = [
    {"info": {"node": "失明", "relationship": "临床表现", "related_node": "糖尿病"}, "score": 0.632},
    {"info": {"node": "再狭窄", "relationship": "临床表现", "related_node": "糖尿病"}, "score": 0.628},
    {"info": {"node": "内疚感", "relationship": "临床表现", "related_node": "糖尿病"}, "score": 0.615},
    {"info": {"node": "创面", "relationship": "临床表现", "related_node": "糖尿病"}, "score": 0.574}
]

grouped_results = group_subgraphs_by_relation(subgraphs)

print(grouped_results)

