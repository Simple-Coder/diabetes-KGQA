"""
Created by xiedong
@Date: 2023/6/5 15:32
"""


class ContextManager:
    def __init__(self):
        self.contexts = {}

    def add_context(self, context_name, context_data):
        self.contexts[context_name] = context_data

    def get_context(self, context_name):
        return self.contexts.get(context_name)

    def remove_context(self, context_name):
        if context_name in self.contexts:
            del self.contexts[context_name]