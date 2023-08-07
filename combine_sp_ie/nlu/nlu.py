"""
Created by xiedong
@Date: 2023/8/7 12:37

输入原始Query，输出Query理解结果

Query理解是KBQA的第一个核心模块，负责对句子的各个成分进行细粒度语义理解，其中两个最重要的模块是
1)实体识别和实体链接，输出问句中有意义的业务相关实体和类型，如商家名称、项目、设施、人群、时间等。
2)依存分析：以分词和词性识别结果为输入，识别问句的主实体、被提问信息、约束等。

实体识别是句法分析的重要步骤，我们先基于序列标注模型识别实体，再链接到数据库中的节点

"""
import re

from combine_sp_ie.models.model_wrapper import ModelService
from combine_sp_ie.nlu.entity_link import EntityLinkService


class NLU():
    def __init__(self):
        self.model_service = ModelService()
        self.entity_link_service = EntityLinkService()

    def query_understanding(self, query):
        # main_entity = None  # 初始化主实体
        # related_entities = []  # 初始化相关实体列表
        # intent = None  # 初始化业务领域
        question_type = None  # 初始化问题类型

        # 1、意图识别、意图强度、ner结果
        intent, intent_conf, ner_result = self.model_service.ner_and_recognize_intent(query)

        # 2、实体链接
        linked_entity_result = self.entity_link_service.entity_links(ner_result)
        # 将所有实体链接结果替换到查询中
        query_with_linked_entities = query
        for linked_entity in linked_entity_result:
            query_with_linked_entities = re.sub(r'\b' + re.escape(linked_entity) + r'\b', linked_entity,
                                                query_with_linked_entities)

        # 3、依存分析
        main_entity, question_info, constraints = self.model_service.dependency_analysis(query_with_linked_entities)

        # main_entity = "故宫"  # 模拟直接识别出主实体
        # domain = "旅游"  # 模拟直接识别出业务领域
        # question_type = "一跳"  # 模拟直接识别出问题类型
        return main_entity, intent, intent_conf, question_type


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
