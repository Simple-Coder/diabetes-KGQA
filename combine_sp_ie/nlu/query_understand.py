"""
Created by xiedong
@Date: 2023/8/8 17:34
"""

from combine_sp_ie.beans.intent_info import IntentInfo
from combine_sp_ie.beans.semantic_info import SemanticInfo
from combine_sp_ie.config.base_config import GlobalConfig
from combine_sp_ie.config.logger_conf import my_log
from combine_sp_ie.models.model_wrapper import model_service
from combine_sp_ie.nlu.entity_link import EntityLinkService

log = my_log.logger


class QueryUnderstand():
    def __init__(self):
        self.entity_link_service = EntityLinkService()

    def query_understanding(self, query):
        semantic_info = SemanticInfo()
        try:
            log.info("nlu 正在识别 query:{}".format(query))
            semantic_info.set_query(query)
            intent_probs, slot_probs = model_service.ner_and_recognize_intent(query)
            # 解析模型识别结果
            all_intents = intent_probs[:GlobalConfig.muti_intent_threshold_num]

            all_intents = self.try_append_default(all_intents)

            all_slots = slot_probs

            log.info("[nlu]理解query:{} 结果:{} {}".format(query, intent_probs, slot_probs))
            for reg_intent in all_intents:
                intent = reg_intent[0]
                intent_intensity = reg_intent[1]
                intent_info = IntentInfo(intent, intent_intensity)
                semantic_info.add_intent_info(intent_info)
            # act 返回对象
            semantic_info.set_entities(all_slots)

            log.info("nlu 识别query:{},结果:{}".format(query, query))
        except Exception as e:
            log.error("nlu 识别query：{} 将返回默认值error:{}".format(query, e))
        return semantic_info

    def try_append_default(self, all_intents):
        default = ('others', 1)
        all_intents.extend([default] * (2 - len(all_intents)))
        return all_intents

# # main_entity = None  # 初始化主实体
# # related_entities = []  # 初始化相关实体列表
# # intent = None  # 初始化业务领域
# question_type = None  # 初始化问题类型
# dependency_analysis = None  # 初始化问题类型
#
# # 1、意图识别、意图强度、ner结果
# intent, intent_conf, ner_result = model_service.ner_and_recognize_intent(query)
#
# # 2、实体链接
# linked_entity_result = self.entity_link_service.entity_links(ner_result)
# # 将所有实体链接结果替换到查询中
# query_with_linked_entities = query
# for linked_entity in linked_entity_result:
#     query_with_linked_entities = re.sub(r'\b' + re.escape(linked_entity) + r'\b', linked_entity,
#                                         query_with_linked_entities)
#
# # 3、依存分析
# main_entity, question_info, constraints = model_service.dependency_analysis(query_with_linked_entities)
#
# # main_entity = "故宫"  # 模拟直接识别出主实体
# # domain = "旅游"  # 模拟直接识别出业务领域
# # question_type = "一跳"  # 模拟直接识别出问题类型
# return main_entity, intent, intent_conf, question_type, dependency_analysis
