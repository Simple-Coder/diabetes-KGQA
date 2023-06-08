"""
Created by xiedong
@Date: 2023/6/6 21:16
"""
from muti_server.utils.logger_conf import my_log
from muti_server.nlg.nlg_config import intent_threshold_config, IntentEnum

log = my_log.logger


class DialogueStateTracker:
    def __init__(self):
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
