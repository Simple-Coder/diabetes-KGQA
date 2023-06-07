"""
Created by xiedong
@Date: 2023/6/6 21:16
"""


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
        if dialog_context:
            dialog_context.set_current_semantic(semantic_info)
