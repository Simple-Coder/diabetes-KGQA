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

    def do_answer_client(self, client, server, answer):
        try:
            server.send_message(client, answer)
        except Exception as e:
            log.error("[nlg] do_answer_client error:{}".format(e))

    def merge_medical_answer(self, answer_info1, answer_info2):
        return ''

    def generate_response(self, client, server, dialog_context):
        current_semantic = dialog_context.get_current_semantic()
        history_sematics = dialog_context.get_history_semantics()

        final_answer_text = self.get_default_answer()
        try:
            # 获取意图
            current_intent_infos = current_semantic.get_intent_infos()
            if len(current_intent_infos) == 0:
                log.error("[nlg] 识别到意图集合为空，将采用默认回答")
                self.do_answer_client(client, server, final_answer_text)
                return

            intent_info1 = current_intent_infos[0]
            intent1 = intent_info1.get_intent()
            strategy1 = intent_info1.get_intent_strategy()
            answer_info1 = intent_info1.get_answer_info()

            intent_info2 = current_intent_infos[1]
            intent2 = intent_info2.get_intent()
            strategy2 = intent_info2.get_intent_strategy()
            answer_info2 = intent_info2.get_answer_info()

            # 2个均为闲聊：取第一个回答
            if strategy1 == IntentEnum.Gossip and strategy2 == IntentEnum.Gossip:
                final_answer_text = self.gossip_robot(intent1)
            # 2个均为诊断：合并答案
            if strategy1 == IntentEnum.Gossip and strategy2 == IntentEnum.Gossip:
                final_answer_text = self.gossip_robot(intent1)

            log.info("nlg 最终回答:{}".format(final_answer_text))
            server.send_message(client, final_answer_text)
        except Exception as e:
            log.error("nlg 生成回答异常:{}".format(e))
            server.send_message(client, 'NLG模块异常啦~~')
        finally:
            dialog_context.add_history_semantic(current_semantic)
