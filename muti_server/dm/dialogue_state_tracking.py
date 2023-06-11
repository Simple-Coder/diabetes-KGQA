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

            intent_info1 = intent_infos[0]
            intent1 = intent_info1.get_intent()
            strategy1 = intent_info1.get_intent_strategy()

            slot_info1 = fill_slot_info(intent1, entities, dialog_context)
            slot_info1 = self.kg_service.search(slot_info1, strategy1)

            if slot_info1:
                intent_info1.set_answer_info(slot_info1)

            intent_info2 = intent_infos[1]
            intent2 = intent_info2.get_intent()
            strategy2 = intent_info2.get_intent_strategy()

            slot_info2 = fill_slot_info(intent2, entities, dialog_context)
            slot_info2 = self.kg_service.search(slot_info2, strategy2)
            if slot_info2:
                intent_info2.set_answer_info(slot_info2)

            dialog_context.set_current_semantic(semantic_info)
            log.info("[dst] update current sematic finish,infos:{}".format(json_str(semantic_info)))
        except Exception as e:
            log.error("[dst] update dialog context error:{}".format(e))
