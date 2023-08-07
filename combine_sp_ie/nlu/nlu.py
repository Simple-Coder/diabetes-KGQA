"""
Created by xiedong
@Date: 2023/8/7 12:37
"""


# nlu.py
def query_understanding(query):
    main_entity = "故宫"  # 模拟直接识别出主实体
    domain = "旅游"  # 模拟直接识别出业务领域
    question_type = "一跳"  # 模拟直接识别出问题类型
    return main_entity, domain, question_type


def relation_recognition(query, domain, syntax_analysis):
    candidate_relations = ["门票", "位置", "游览", "时间", "价格"]  # 假设的候选关系
    relation_scores = {}  # 存储关系及其分数的字典

    # 基于依存分析等信息，对候选关系进行评分
    for relation in candidate_relations:
        score = calculate_relation_score(query, domain, relation, syntax_analysis)
        relation_scores[relation] = score

    # 根据分数对关系进行排序
    sorted_relations = sorted(relation_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_relations


def calculate_relation_score(query, domain, relation, syntax_analysis):
    # 实现计算关系得分逻辑
    pass
