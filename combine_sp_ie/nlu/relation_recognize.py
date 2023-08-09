"""
Created by xiedong
@Date: 2023/8/8 17:36
"""
from combine_sp_ie.models.relation_matching_model import RelationMatchingModel


class RelationRecognize():
    def __init__(self, args):
        self.args = args
        self.relation_match_model = RelationMatchingModel(args.bert_dir)

    def relation_recognition(self, query, domain, syntax_analysis):
        candidate_relations = ["门票", "位置", "游览", "时间", "价格"]  # 假设的候选关系
        relation_scores = {}  # 存储关系及其分数的字典

        # 基于依存分析等信息，对候选关系进行评分
        for relation in candidate_relations:
            score = self.calculate_relation_score(query, domain, relation, syntax_analysis)
            relation_scores[relation] = score

        # 根据分数对关系进行排序
        sorted_relations = sorted(relation_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_relations

    def calculate_relation_score(self, query, domain, relation, syntax_analysis):
        return 0.9
