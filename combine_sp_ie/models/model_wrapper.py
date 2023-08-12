"""
Created by xiedong
@Date: 2023/8/7 15:35
模型包装类
"""
from ltp import LTP
from transformers import BertTokenizer, BertModel
from combine_sp_ie.config.base_config import GlobalConfig


class ModelService():
    def __init__(self):
        self.ltp_model = LTP()
        self.tokenizer = BertTokenizer.from_pretrained(GlobalConfig.bert_dir)
        self.model = BertModel.from_pretrained(GlobalConfig.bert_dir)

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
        ner_result = ["糖尿病"]
        return intent, intent_conf, ner_result

    def dependency_analysis(self, query):
        return '', '', ''


model_service = ModelService()
