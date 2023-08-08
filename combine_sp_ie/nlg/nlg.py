"""
Created by xiedong
@Date: 2023/8/7 12:43
"""
from combine_sp_ie.config.chat_config import semantic_slot


class NLG():
    def __init__(self, args):
        self.args = args

    def generate_response(self, query, subgraph):
        # check
        if not subgraph:
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
