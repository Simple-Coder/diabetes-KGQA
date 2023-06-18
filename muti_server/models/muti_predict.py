"""
Created by xiedong
@Date: 2023/6/7 12:35
"""
from transformers import logging

from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_trainer import Trainer
from muti_server.utils.logger_conf import my_log

log = my_log.logger

logging.set_verbosity_error()


class MutiPredictWrapper:
    def __init__(self, args):
        self.args = args
        self.trainer = Trainer(args)

    def predict(self, text):
        log.info("[model] predict text:{}".format(text))
        return self.trainer.predict(text)


if __name__ == '__main__':
    args = ModelConfig()
    predictor = MutiPredictWrapper(args)
    input_text = "请问糖尿病有什么症状,病因是什么"
    intent_probs, slot_probs = predictor.predict(input_text)

    print("Intent probabilities:", sorted(intent_probs, key=lambda x: x[1]))
    print("Slot probabilities:", slot_probs)
