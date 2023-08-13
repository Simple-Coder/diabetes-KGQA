"""
Created by xiedong
@Date: 2023/8/7 15:35
模型包装类
"""
from ltp import LTP
from transformers import BertTokenizer, BertModel
from combine_sp_ie.config.base_config import GlobalConfig
from transformers import logging
from combine_sp_ie.config.logger_conf import my_log
from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_trainer import Trainer

logging.set_verbosity_error()
log = my_log.logger


class ModelService():
    def __init__(self):
        args = ModelConfig()
        self.ltp_model = LTP()
        self.tokenizer = BertTokenizer.from_pretrained(GlobalConfig.bert_dir)
        self.model = BertModel.from_pretrained(GlobalConfig.bert_dir)
        self.trainer = Trainer(args)

    def ner(self, query):
        result = self.model.pipeline([query], tasks=["cws", "ner"])
        return result

    def ner_and_recognize_intent(self, query):
        """
        根据query 做ner 与 intent识别
        :param query:
        :return:
        """
        log.info("[model] predict text:{}".format(query))
        return self.trainer.predict(query)
        # intent = 'Reason_Disease'
        # intent_conf = 0.5
        # ner_result = ["糖尿病"]
        # return intent, intent_conf, ner_result

    def dependency_analysis(self, query):
        return '', '', ''


model_service = ModelService()

if __name__ == '__main__':
    predictor = model_service
    input_text = "请问糖尿病有什么症状,病因是什么"
    intent_probs, slot_probs = predictor.ner_and_recognize_intent(input_text)

    print("Intent probabilities:", sorted(intent_probs, key=lambda x: x[1]))
    print("Slot probabilities:", slot_probs)
