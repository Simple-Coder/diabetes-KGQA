"""
Created by xiedong
@Date: 2023/8/8 9:29
"""


# class ResponseGenerator:
#     def generate_response(self, query, subgraph):
#         if not subgraph:
#             return "抱歉，未找到相关信息。"
#
#         response_template = self.select_response_template(subgraph)
#         filled_response = self.fill_response_template(response_template, subgraph)
#
#         return filled_response
#
#     def select_response_template(self, subgraph):
#         if "门票" in subgraph:
#             return "故宫的门票价格是{}。"
#         elif "位置" in subgraph:
#             return "故宫位于{}。"
#         else:
#             return "关于{}的信息是{}。"
#
#     def fill_response_template(self, template, subgraph):
#         # 根据子图内容填充回复模板
#         filled_response = template.format(*subgraph[1:])  # 忽略第一个元素（主题实体）
#         return filled_response
#
#
# # 示例使用
# response_generator = ResponseGenerator()
#
# # 假设查询结果的子图
# subgraph = ["故宫", "门票价格", "100元"]
#
# response = response_generator.generate_response("故宫门票价格是多少？", subgraph)
# print(response)

class ResponseGenerator:
    def generate_response(self, query, subgraph):
        if not subgraph:
            return "抱歉，未找到相关信息。"

        response_template = self.select_response_template(subgraph)
        filled_response = self.fill_response_template(response_template, subgraph)

        return filled_response

    def select_response_template(self, subgraph):
        if "门票" in subgraph:
            return "故宫的门票价格是{}。"
        elif "位置" in subgraph:
            return "故宫位于{}。"
        elif "症状" in subgraph:
            return "糖尿病的症状包括{}。"
        else:
            return "关于{}的信息是{}。"

    def fill_response_template(self, template, subgraph):
        # 根据子图内容填充回复模板
        if isinstance(subgraph[-1], list):
            filled_response = template.format(", ".join(subgraph[-1]))
        else:
            filled_response = template.format(*subgraph[1:])  # 忽略第一个元素（主题实体）
        return filled_response


# 示例使用
response_generator = ResponseGenerator()

# 修改后的子图
subgraph = ["糖尿病", "症状", ["吃的多", "喝得多", "尿量多"]]

response = response_generator.generate_response("糖尿病有哪些症状？", subgraph)
print(response)
