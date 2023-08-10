"""
Created by xiedong
@Date: 2023/8/7 12:43
"""
from combine_sp_ie.config.chat_config import semantic_slot

from combine_sp_ie.config.logger_conf import my_log

log = my_log.logger


class NLG():
    def __init__(self, args):
        self.args = args

    def group_subgraphs_by_relation(self, subgraphs):
        if not subgraphs or len(subgraphs) == 0:
            log.warn("[nlg]子图为空，合并子图结束,ext")
            return None
        grouped_subgraphs = {}  # 创建一个空的字典用于分组

        for graph in subgraphs:
            relation_info = graph["info"]["relationship"]
            related_node = graph["info"]["related_node"]

            key = (relation_info, related_node)  # 使用关系和关联节点作为组合的键
            if key not in grouped_subgraphs:
                grouped_subgraphs[key] = {"relation": relation_info, "related_node": related_node, "subgraphs": []}

            grouped_subgraphs[key]["subgraphs"].append(graph)

        grouped_results = list(grouped_subgraphs.values())
        return grouped_results

    def generate_response(self, query, subgraph):
        # check
        if not subgraph:
            log.warn("【NLG】子图不存在，返回默认回复")
            return semantic_slot.get("others").get("replay_answer")

        # 1、根据 {子图关系} 查询回复模板
        response_template = self.select_response_template(subgraph)
        # 2、填充 回复模板
        filled_response = self.fill_response_template(response_template, subgraph)

        return filled_response

    def select_response_template(self, subgraph):
        relation = subgraph[1]
        response_info = semantic_slot.get(relation)
        if not response_info:
            return semantic_slot.get("others").get("replay_answer")

        return response_info.get("response_template")

    def fill_response_template(self, template, subgraph):
        # 根据子图内容填充回复模板
        if isinstance(subgraph[-1], list):
            filled_response = template.format(", ".join(subgraph[-1]))
        else:
            filled_response = template.format(*subgraph[1:])  # 忽略第一个元素（主题实体）
        return filled_response
