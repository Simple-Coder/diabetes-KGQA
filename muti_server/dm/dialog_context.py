"""
Created by xiedong
@Date: 2023/6/5 21:06
"""


class DialogContext:
    def __init__(self, client_id):
        self.client_id = client_id
        self.current_semantic = None  # 语义信息
        self.current_slot_infos = []

        self.history_semantics = []
        self.history_slot_infos = []

        self.context_data = {}  # 上下文数据
        self.pre_dialog_context = None

    def add_current_slot_info(self, slot_info):
        self.current_slot_infos.append(slot_info)

    def get_current_slot_infos(self):
        return self.current_slot_infos

    def set_current_slot_infos(self, slot_infos):
        self.current_slot_infos = slot_infos

    def add_history_semantic(self, sematic_info):
        self.history_semantics.append(sematic_info)

    def get_history_semantics(self):
        return self.history_semantics

    def add_history_slot_info(self, slot_info):
        self.history_slot_infos.append(slot_info)

    def get_history_slot_infos(self):
        return self.history_slot_infos

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
