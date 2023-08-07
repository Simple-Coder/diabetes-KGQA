"""
Created by xiedong
@Date: 2023/8/7 13:06
"""
from ltp import LTP

class DependencyAnalysis:
    def __init__(self):
        self.ltp = LTP()

    def analyze_dependency(self, query):
        result = self.ltp.pipeline([query], tasks=["cws", "dep"])
        dep_parse = result[1][0]  # 获取依存分析结果

        words = result[0][0]

        main_entity = None
        question_info = None
        constraints = []

        for idx, (head, label) in enumerate(zip(dep_parse['head'], dep_parse['label'])):
            word = words[idx]
            if label == 'HED':  # 根节点为问题的谓词，一般为主谓关系
                main_entity = word
            elif label == 'SBV':  # 主谓关系，名词作为问题的被提问信息
                question_info = word
            elif label == 'ATT':  # 修饰词作为约束条件
                constraints.append(word)

        return main_entity, question_info, constraints

# 实例化依存分析类
dep_analyzer = DependencyAnalysis()

# 进行依存分析
query = "故宫周末有学生票吗"
main_entity, question_info, constraints = dep_analyzer.analyze_dependency(query)

# 打印分析结果
print("主实体:", main_entity)
print("被提问信息:", question_info)
print("约束条件:", constraints)
