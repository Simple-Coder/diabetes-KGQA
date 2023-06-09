"""
Created by xiedong
@Date: 2023/6/6 21:52
"""
from muti_server.nlg.nlg_config import *
import random
from muti_server.utils.logger_conf import my_log

log = my_log.logger


class NLG():
    def __init__(self, args):
        self.args = args

    def gossip_robot(self, intent):
        return random.choice(gossip_corpus.get(intent))

    def get_default_answer(self):
        return semantic_slot.get("others")["replay_answer"]

    def generate_response(self, client, server, dialog_context):
        try:
            # 获取意图
            intent_infos = dialog_context.get_current_semantic().get_intent_infos()
            intent_info1 = intent_infos[0]
            intent1 = intent_info1.get_intent()
            strategy1 = intent_info1.get_intent_strategy()

            if strategy1 == IntentEnum.Gossip:
                answer = self.gossip_robot(intent1)
            else:
                answer = self.get_default_answer()
            # 回答不知道
            log.info("nlg 最终回答:{}".format(answer))
            server.send_message(client, answer)
        except Exception as e:
            log.error("nlg 生成回答异常:{}".format(e))
            server.send_message(client, 'NLG模块异常啦~~')
