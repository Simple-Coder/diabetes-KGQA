"""
Created by xiedong
@Date: 2023/8/7 15:35
模型包装类
"""
from ltp import LTP


class ModelService():
    def __init__(self):
        self.model = LTP()

    def ner(self, query):
        result = self.model.pipeline([query], tasks=["cws", "ner"])
        return result

    def ner_and_recognize_intent(self, query):
        """
        根据query 做ner 与 intent识别
        :param query:
        :return:
        """
        intent = 'Reason_Disease'
        intent_conf = 0.5
        ner_result = []
        return intent, intent_conf, ner_result

    def dependency_analysis(self, query):
        pass
