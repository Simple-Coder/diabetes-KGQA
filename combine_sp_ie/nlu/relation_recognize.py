"""
Created by xiedong
@Date: 2023/8/8 17:36
"""
from combine_sp_ie.models.relation_matching_model import relation_matching_model
from combine_sp_ie.models.model_wrapper import model_service
from combine_sp_ie.utils.relation import relations_map_service


class RelationRecognize():
    def relation_recognition(self, semantic_info):
        # candidate_relations = relations_map_service.get_cn_relations()  # 候选关系
        # relation_scores = {}  # 存储关系及其分数的字典
        # #
        # # # 基于依存分析等信息，对候选关系进行评分
        # for relation in candidate_relations:
        #     score = self.calculate_relation_score(query, relation, syntax_analysis)
        #     relation_scores[relation] = score
        #
        # # 根据分数对关系进行排序
        # sorted_relations = sorted(relation_scores.items(), key=lambda x: x[1], reverse=True)
        # return sorted_relations
        return semantic_info

    def calculate_relation_score(self, query, domain, relation, syntax_analysis):
        return 0.9
