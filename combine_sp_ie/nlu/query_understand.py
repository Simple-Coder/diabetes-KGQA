"""
Created by xiedong
@Date: 2023/8/8 17:34
"""

import re

from combine_sp_ie.config.logger_conf import my_log
from combine_sp_ie.models.model_wrapper import model_service
from combine_sp_ie.nlu.entity_link import EntityLinkService

log = my_log.logger


class QueryUnderstand():
    def __init__(self):
        self.entity_link_service = EntityLinkService()

    def query_understanding(self, query):
        # main_entity = None  # 初始化主实体
        # related_entities = []  # 初始化相关实体列表
        # intent = None  # 初始化业务领域
        question_type = None  # 初始化问题类型
        dependency_analysis = None  # 初始化问题类型

        # 1、意图识别、意图强度、ner结果
        intent, intent_conf, ner_result = model_service.ner_and_recognize_intent(query)

        # 2、实体链接
        linked_entity_result = self.entity_link_service.entity_links(ner_result)
        # 将所有实体链接结果替换到查询中
        query_with_linked_entities = query
        for linked_entity in linked_entity_result:
            query_with_linked_entities = re.sub(r'\b' + re.escape(linked_entity) + r'\b', linked_entity,
                                                query_with_linked_entities)

        # 3、依存分析
        main_entity, question_info, constraints = model_service.dependency_analysis(query_with_linked_entities)

        # main_entity = "故宫"  # 模拟直接识别出主实体
        # domain = "旅游"  # 模拟直接识别出业务领域
        # question_type = "一跳"  # 模拟直接识别出问题类型
        return main_entity, intent, intent_conf, question_type, dependency_analysis
