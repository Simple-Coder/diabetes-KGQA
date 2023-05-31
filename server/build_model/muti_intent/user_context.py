"""
Created by xiedong
@Date: 2023/5/30 21:20
"""


class UserContext:
    def __init__(self, clientId, username, query):
        self.clientId = clientId
        self.username = username
        self.query = query
        self.all_slots = []
        self.all_intents = []
        self.slot_info = None

    def get_username(self):
        return self.username

    def get_context_slot(self):
        return self.slot_info

    def set_context_slot(self, slot_info):
        self.slot_info = slot_info

    def get_query(self):
        return self.query

    def set_username(self, username):
        self.username = username

    def set_all_slots(self, all_slots):
        self.all_slots = all_slots

    def get_all_slots(self):
        return self.all_slots

    def set_all_intents(self, all_intents):
        self.all_intents = all_intents

    def get_all_intents(self):
        return self.all_intents
