"""
Created by xiedong
@Date: 2023/8/7 12:43
"""


class NLG():
    def __init__(self, args):
        self.args = args

    def generate_response(self, query, subgraph):
        if not subgraph:
            return "抱歉，未找到相关信息。"
        # 1、根据 {子图关系} 查询回复模板
        response_template = self.select_response_template(subgraph)
        # 2、填充 回复模板
        filled_response = self.fill_response_template(response_template, subgraph)

        return filled_response

    def select_response_template(self, subgraph):
        if "门票" in subgraph:
            return "故宫的门票价格是{}。"
        elif "位置" in subgraph:
            return "故宫位于{}。"
        else:
            return "关于{}的信息是{}。"

    def fill_response_template(self, template, subgraph):
        # 根据子图内容填充回复模板
        filled_response = template.format(*subgraph[1:])  # 忽略第一个元素（主题实体）
        return filled_response
