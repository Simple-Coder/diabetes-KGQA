"""
Created by xiedong
@Date: 2023/8/13 11:55
"""


class SemanticInfo(object):
    def __init__(self):
        self.intent_infos = []
        self.entities = None
        self.query = None
        self.answer_sub_graphs = None

    def add_intent_info(self, intent_info):
        self.intent_infos.append(intent_info)

    def get_intent_infos(self):
        return self.intent_infos

    def set_entities(self, entities):
        self.entities = entities

    def get_entities(self):
        return self.entities

    def set_query(self, query):
        self.query = query

    def get_query(self):
        return self.query

    def set_answer_sub_graphs(self, answer_sub_graphs):
        self.answer_sub_graphs = answer_sub_graphs

    def get_answer_sub_graphs(self):
        return self.answer_sub_graphs
