"""
Created by xiedong
@Date: 2023/6/5 15:32
"""

from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_predict import MutiPredictWrapper
from muti_server.utils.logger_conf import my_log
# from muti_server.nlu.nlu_utils import build_intent_strategy, build_intent_enum
from muti_server.nlu.nlu_utils import build_intent_enum, get_white_multi_hop_nlu
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
            semantic_info.set_query(text)
            # predict_result = self.predictor.predict(text)
            intent_probs, slot_probs = self.predictor.predict(text)

            # 解析模型识别结果
            all_intents = intent_probs[:self.model_config.muti_intent_threshold_num]

            # TODO:处理多跳白名单识别
            all_intents, slot_probs = get_white_multi_hop_nlu(text, all_intents, slot_probs)

            all_intents = self.try_append_default(all_intents)

            all_slots = slot_probs

            log.info("[nlu]理解query:{} 结果:all_intents:{},all_slots:{}".format(text, json_str(all_intents),
                                                                             json_str(all_slots)))
            for reg_intent in all_intents:
                intent = reg_intent[0]
                intent_intensity = reg_intent[1]
                intent_hop = 1
                if len(reg_intent) > 2:
                    intent_hop = reg_intent[2]

                intent_info = IntentInfo(intent, intent_intensity, intent_hop)
                semantic_info.add_intent_info(intent_info)
            # act 返回对象
            semantic_info.set_entities(all_slots)

            log.info("nlu 识别query:{},结果:{}".format(text, json_str(semantic_info)))
        except Exception as e:
            log.error("nlu 识别query：{} 将返回默认值error:{}".format(text, e))
        return semantic_info

    def try_append_default(self, all_intents):
        default = ('others', 1)
        all_intents.extend([default] * (2 - len(all_intents)))
        return all_intents


# 意图信息
class IntentInfo:
    def __init__(self, intent, intensity, intent_hop=1):
        """
        :param intent: 意图 如：greep、goodbye等
        :param intensity: 意图强度  例如：0.6
        """
        self.intent = intent
        self.intensity = intensity
        self.intent_hop = intent_hop
        self.intent_enum = build_intent_enum(self.intent, intensity)
        # self.intent_strategy = build_intent_strategy(self.intent, self.intensity)
        self.answer_info = None

    # def get_intent_strategy(self):
    #     return self.intent_strategy

    def get_intent(self):
        return self.intent

    def get_intensity(self):
        return self.intensity

    def get_intent_hop(self):
        return self.intent_hop

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
