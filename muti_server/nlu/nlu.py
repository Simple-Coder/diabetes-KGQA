"""
Created by xiedong
@Date: 2023/6/5 15:32
"""

from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_predict import MutiPredictWrapper
from muti_server.utils.logger_conf import my_log
from muti_server.nlu.nlu_utils import build_intent_strategy, build_intent_enum
from muti_server.utils.json_utils import json_str

log = my_log.logger


# 自然语言理解（NLU）模块
class NLU:
    def __init__(self, args):
        self.run_args = args
        self.model_config = ModelConfig()
        # 包装预测
        self.predictor = MutiPredictWrapper(self.model_config)

    def predict(self, text):
        """
        预测输入文本
        :param text: 例如：糖尿病有什么症状啊
        :return: 意图,槽位  例如：[(greep,0.6),(goodbye,0.5)],(槽位)
        """
        semantic_info = SemanticInfo()
        try:
            log.info("nlu 正在识别 query:{}".format(text))
            # predict_result = self.predictor.predict(text)
            intent_probs, slot_probs = self.predictor.predict(text)

            # 解析模型识别结果
            all_intents = intent_probs[:self.model_config.muti_intent_threshold_num]
            all_slots = slot_probs

            intent_info1 = all_intents[0]
            intent1 = intent_info1[0]
            intent1_intensity = intent_info1[1]
            intent_info1 = IntentInfo(intent1, intent1_intensity)
            # intent_info1.set_intent_enum(intent1)

            intent_info2 = all_intents[1]
            intent2 = intent_info2[0]
            intent2_intensity = intent_info2[1]
            intent_info2 = IntentInfo(intent2, intent2_intensity)
            # intent_info2.set_intent_enum(intent2)

            # act 返回对象
            semantic_info.set_entities(all_slots)
            semantic_info.add_intent_info(intent_info1)
            semantic_info.add_intent_info(intent_info2)

            log.info("nlu 识别query:{},结果:{}".format(text, json_str(semantic_info)))
        except Exception as e:
            log.error("nlu 识别query：{} 将返回默认值error:{}".format(text, e))
        return semantic_info


# 意图信息
class IntentInfo:
    def __init__(self, intent, intensity):
        """
        :param intent: 意图 如：greep、goodbye等
        :param intensity: 意图强度  例如：0.6
        """
        self.intent = intent
        self.intent_enum = build_intent_enum(self.intent)
        self.intensity = intensity
        self.intent_strategy = build_intent_strategy(self.intent, self.intensity)
        self.answer_info = None

    def get_intent_strategy(self):
        return self.intent_strategy

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


class SemanticInfo(object):
    def __init__(self):
        self.intent_infos = []
        self.entities = None

    def add_intent_info(self, intent_info):
        self.intent_infos.append(intent_info)

    def get_intent_infos(self):
        return self.intent_infos

    def set_entities(self, entities):
        self.entities = entities

    def get_entities(self):
        return self.entities
