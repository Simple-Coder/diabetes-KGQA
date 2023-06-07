"""
Created by xiedong
@Date: 2023/6/5 21:06
"""


class DialogContext:
    def __init__(self, client_id):
        self.client_id = client_id
        self.current_semantic = None  # 语义信息
        self.context_data = {}  # 上下文数据

    def set_current_semantic(self, sematic_info):
        self.current_semantic = sematic_info

    def get_current_semantic(self):
        return self.current_semantic

    def add_context_data(self, key, value):
        self.context_data[key] = value

    def get_context_data(self, key):
        return self.context_data.get(key)

    def remove_context_data(self, key):
        if key in self.context_data:
            del self.context_data[key]
