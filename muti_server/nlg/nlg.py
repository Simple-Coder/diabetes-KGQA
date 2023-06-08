"""
Created by xiedong
@Date: 2023/6/6 21:52
"""
from muti_server.nlg.nlg_config import *
import random
from muti_server.utils.logger_conf import my_log

log = my_log.logger


class NLG():
    def gossip_robot(self, intent):
        return random.choice(gossip_corpus.get(intent))

    def get_default_answer(self):
        return semantic_slot.get("others")["replay_answer"]

    def generate_response(self, client, server, dialog_context):
        try:
            # 获取意图
            intent_infos = dialog_context.get_current_semantic().get_intent_infos()
            intent_info1 = intent_infos[0]
            intent = intent_info1.get_intent()
            answer = self.get_default_answer()

            if intent in gossip_corpus.keys():
                server.send_message(client, self.gossip_robot(intent))
            else:
                # 回答不知道
                server.send_message(client, self.get_default_answer())
        except Exception as e:
            log.error("nlg 生成回答异常:{}".format(e))
            server.send_message(client, 'NLG模块异常啦~~')
