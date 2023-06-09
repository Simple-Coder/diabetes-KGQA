"""
Created by xiedong
@Date: 2023/6/6 21:16
"""
from muti_server.utils.logger_conf import my_log
from muti_server.nlg.nlg_utils import fill_slot_info
from muti_server.knowledge_graph.service import KgService

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
        dialog_context = self.contexts.get(client_id)
        if not dialog_context:
            return
        dialog_context.set_current_semantic(semantic_info)

        intent_infos = dialog_context.get_current_semantic().get_intent_infos()
        entities = dialog_context.get_current_semantic().get_entities()

        intent_info1 = intent_infos[0]
        intent1 = intent_info1.get_intent()
        strategy1 = intent_info1.get_intent_strategy()

        slot_info1 = fill_slot_info(intent1, entities)
        slot_info1 = self.kg_service.search(slot_info1, strategy1)

        if slot_info1:
            # TODO:存储
            pass

        intent_info2 = intent_infos[1]
        intent2 = intent_info2.get_intent()
        strategy2 = intent_info2.get_intent_strategy()

        slot_info2 = fill_slot_info(intent2, entities)
        slot_info2 = self.kg_service.search(slot_info2, strategy2)
        if slot_info2:
            # TODO:存储
            pass
