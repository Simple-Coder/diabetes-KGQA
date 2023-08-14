"""
Created by xiedong
@Date: 2023/8/7 15:35
模型包装类
"""
from ltp import LTP
import torch
import torch.nn as nn
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

    def calculate_similarity_scores(self, question, candidate_texts):
        question_tokens = self.tokenizer.tokenize(question)
        question_inputs = self.tokenizer.encode_plus(question_tokens, add_special_tokens=True, return_tensors='pt')
        question_embedding = self.model(**question_inputs).last_hidden_state.mean(dim=1)  # 使用平均池化

        candidate_embeddings = []
        for text in candidate_texts:
            text_tokens = self.tokenizer.tokenize(text)
            text_inputs = self.tokenizer.encode_plus(text_tokens, add_special_tokens=True, return_tensors='pt')
            text_embedding = self.model(**text_inputs).last_hidden_state.mean(dim=1)
            candidate_embeddings.append(text_embedding)

        similarity_scores = []
        for embedding in candidate_embeddings:
            # 计算余弦相似度得分
            cos_sim = torch.nn.functional.cosine_similarity(question_embedding, embedding, dim=1)
            similarity_scores.append(cos_sim.item())  # 将相似度得分添加到列表中

        return similarity_scores


model_service = ModelService()

if __name__ == '__main__':
    predictor = model_service
    input_text = "请问糖尿病有什么症状,病因是什么"
    intent_probs, slot_probs = predictor.ner_and_recognize_intent(input_text)

    print("Intent probabilities:", sorted(intent_probs, key=lambda x: x[1]))
    print("Slot probabilities:", slot_probs)
