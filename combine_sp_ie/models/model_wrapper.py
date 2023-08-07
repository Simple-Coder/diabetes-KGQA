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
