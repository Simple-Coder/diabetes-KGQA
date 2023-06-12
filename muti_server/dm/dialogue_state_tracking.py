"""
Created by xiedong
@Date: 2023/6/6 21:16
"""
from muti_server.utils.logger_conf import my_log
from muti_server.nlg.nlg_utils import fill_slot_info
from muti_server.knowledge_graph.service import KgService
from muti_server.utils.json_utils import json_str

log = my_log.logger


class DialogueStateTracker:
    def __init__(self, args):
        self.args = args
        self.kg_service = KgService(args)
        self.contexts = {}

    def add_context(self, context_name, context_data):
        self.contexts[context_name] = context_data

    def get_context(self, context_name):
        return self.contexts.get(context_name)

    def remove_context(self, context_name):
        if context_name in self.contexts:
            del self.contexts[context_name]

    def update_context_sematic_info(self, client_id, semantic_info):
        try:
            """
                    记录语义信息到上下文
                    填槽
                    查询neo4j记录每个意图的答案
                    :param client_id:
                    :param semantic_info:
                    :return:
                    """
            dialog_context = self.contexts.get(client_id)
            if not dialog_context:
                return

            intent_infos = semantic_info.get_intent_infos()
            entities = semantic_info.get_entities()

            if len(intent_infos) == 0:
                log.error("[dm] 意图识别为空，不处理查询neo4j")
                return

            for intent_info in intent_infos:
                intent = intent_info.get_intent()
                strategy = intent_info.get_intent_strategy()
                slot_info = fill_slot_info(intent, entities, dialog_context)
                slot_info = self.kg_service.search(slot_info, strategy)

                if slot_info:
                    intent_info.set_answer_info(slot_info)

            dialog_context.set_current_semantic(semantic_info)
            log.info("[dst] update current sematic finish,infos:{}".format(json_str(semantic_info)))
        except Exception as e:
            log.error("[dst] update dialog context error:{}".format(e))
