"""
Created by xiedong
@Date: 2023/8/13 12:02
"""

from combine_sp_ie.nlu.nlu_utils import build_intent_enum


class IntentInfo:
    def __init__(self, intent, intensity):
        """
        :param intent: 意图 如：greep、goodbye等
        :param intensity: 意图强度  例如：0.6
        """
        self.intent = intent
        self.intensity = intensity
        self.intent_enum = build_intent_enum(self.intent, intensity)
        # self.intent_strategy = build_intent_strategy(self.intent, self.intensity)
        self.answer_info = None

    # def get_intent_strategy(self):
    #     return self.intent_strategy

    def get_intent(self):
        return self.intent

    def get_intensity(self):
        return self.intensity

    def get_intent_enum(self):
        return self.intent_enum

    def get_answer_info(self):
        return self.answer_info

    def set_answer_info(self, answer_info):
        self.answer_info = answer_info
