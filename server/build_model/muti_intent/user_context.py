"""
Created by xiedong
@Date: 2023/5/30 21:20
"""
from muti_chat_config import AnswerEnum


class UserContext:
    def __init__(self, clientId, username, query):
        self.clientId = clientId
        self.username = username
        self.query = query
        self.all_slots = []
        self.all_intents = []
        self.slot_info = None
        self.answer_enum = AnswerEnum.ANSWER_ALL_OTHERS.ANSWER_ALL_OTHERS
        self.answer_info1 = None
        self.answer_info2 = None

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

    def get_answer_enum(self):
        return self.answer_enum

    def set_answer_enum(self, answer_enum):
        self.answer_enum = answer_enum

    def get_answer_info1(self):
        return self.answer_info1

    def set_answer_info1(self, answer_info1):
        self.answer_info1 = answer_info1

    def get_answer_info2(self):
        return self.answer_info2

    def set_answer_info2(self, answer_info2):
        self.answer_info2 = answer_info2
