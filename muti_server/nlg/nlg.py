"""
Created by xiedong
@Date: 2023/6/6 21:52
"""
from muti_server.nlg.nlg_config import *
import random
from muti_server.utils.logger_conf import my_log
from muti_server.utils.json_utils import json_str

log = my_log.logger


class NLG():
    def __init__(self, args):
        self.args = args

    def handle_gossip(self, intent, client, server):
        answer = random.choice(gossip_corpus.get(intent))
        self.do_answer_client(client, server, answer)

    def get_default_answer(self):
        return semantic_slot.get("others")["replay_answer"]

    def do_answer_client(self, client, server, answer):
        try:
            server.send_message(client, answer)
        except Exception as e:
            log.error("[nlg] do_answer_client error:{}".format(e))

    def generate_response(self, client, server, dialog_context):
        current_semantic = dialog_context.get_current_semantic()
        history_sematics = dialog_context.get_history_semantics()

        try:
            if not current_semantic:
                log.error("[nlg] 当前语义识别为null，将采用默认回答")
                self.do_answer_client(client, server, self.get_default_answer())
                return

            # 获取意图
            current_intent_infos = current_semantic.get_intent_infos()
            if len(current_intent_infos) == 0:
                log.error("[nlg] 识别到意图集合为空，将采用默认回答")
                self.do_answer_client(client, server, self.get_default_answer())
                return

            intent_info1 = current_intent_infos[0]
            intent1 = intent_info1.get_intent()
            intent_enum1 = intent_info1.get_intent_enum()
            answer_info1 = intent_info1.get_answer_info()

            intent_info2 = current_intent_infos[1]
            intent2 = intent_info2.get_intent()
            intent_enum2 = intent_info2.get_intent_enum()
            answer_info2 = intent_info2.get_answer_info()

            # 2个均为闲聊：取第一个回答
            if intent_enum1 == IntentEnum.Gossip and intent_enum2 == IntentEnum.Gossip:
                self.handle_gossip(intent1, client, server)
            # 2个均为诊断：合并答案
            elif intent_enum1 == IntentEnum.Medical and intent_enum2 == IntentEnum.Medical:
                self.handle_all_medical(client, server, answer_info1, answer_info2)
            # 2个均为澄清：合并澄清
            elif intent_enum1 == IntentEnum.Clarify and intent_enum2 == IntentEnum.Clarify:
                self.handle_gossip(intent1, client, server)
            # 2个均为未知：返回学习中
            elif intent_enum1 == IntentEnum.Others and intent_enum2 == IntentEnum.Others:
                self.handle_gossip(intent1, client, server)
            # 1个闲聊，一个诊断：合并诊断
            elif intent_enum1 == IntentEnum.Gossip and intent_enum2 == IntentEnum.Medical:
                self.handle_gossip(intent1, client, server)
                # 2个均为澄清：合并澄清
            elif intent_enum1 == IntentEnum.Clarify and intent_enum2 == IntentEnum.Clarify:
                self.handle_gossip(intent1, client, server)
                # 1个闲聊，一个澄清：合并澄清
            elif intent_enum1 == IntentEnum.Gossip and intent_enum2 == IntentEnum.Clarify:
                self.handle_gossip(intent1, client, server)
                # 1个闲聊，一个未知：优先闲聊
            elif intent_enum1 == IntentEnum.Gossip and intent_enum2 == IntentEnum.Others:
                self.handle_gossip(intent1, client, server)
                # 1个诊断，一个闲聊：只处理诊断
            elif intent_enum1 == IntentEnum.Medical and intent_enum2 == IntentEnum.Gossip:
                self.handle_gossip(intent1, client, server)
                # 1个诊断，一个澄清：先诊断，再澄清
            elif intent_enum1 == IntentEnum.Medical and intent_enum2 == IntentEnum.Clarify:
                self.handle_gossip(intent1, client, server)
                # 1个诊断，一个未知：只诊断
            elif intent_enum1 == IntentEnum.Medical and intent_enum2 == IntentEnum.Others:
                final_answer_text = self.handle_medical(client, server, answer_info1)
                # 一个澄清，一个闲聊：只处理澄清
            elif intent_enum1 == IntentEnum.Clarify and intent_enum2 == IntentEnum.Gossip:
                self.handle_gossip(intent1, client, server)
                # 一个澄清，一个诊断：合并诊断后澄清
            elif intent_enum1 == IntentEnum.Clarify and intent_enum2 == IntentEnum.Medical:
                self.handle_gossip(intent1, client, server)
                # 一个澄清，一个未知：只处理澄清
            elif intent_enum1 == IntentEnum.Clarify and intent_enum2 == IntentEnum.Others:
                self.handle_gossip(intent1, client, server)
                # 一个未知，一个闲聊：学习中
            elif intent_enum1 == IntentEnum.Others and intent_enum2 == IntentEnum.Gossip:
                self.handle_gossip(intent1, client, server)
                # 一个未知，一个澄清：学习中
            elif intent_enum1 == IntentEnum.Others and intent_enum2 == IntentEnum.Clarify:
                self.handle_gossip(intent1, client, server)
                # 一个未知，一个诊断：学习中
            elif intent_enum1 == IntentEnum.Others and intent_enum2 == IntentEnum.Medical:
                self.handle_gossip(intent1, client, server)
                # 1为接受，找上下文回答
            elif intent_enum1 == IntentEnum.Accept:
                self.handle_gossip(intent1, client, server)
                # 2为接受，找上下文回答
            elif intent_enum2 == IntentEnum.Clarify:
                self.handle_gossip(intent1, client, server)
            else:
                log.info("[nlg]未识别到回答策略，intent1:{},intent2:{}".format(json_str(intent_info1),
                                                                              json_str(intent_info2)))
        except Exception as e:
            log.error("nlg 生成回答异常:{}".format(e))
            server.send_message(client, 'NLG模块异常啦~~')
        finally:
            dialog_context.add_history_semantic(current_semantic)

    def handle_medical(self, client, server, answer_info1):
        if not answer_info1:
            return self.get_default_answer()
        return answer_info1.get('replay_answer')

    def handle_all_medical(self, client, server, answer_info1, answer_info2):
        default_answer = self.get_default_answer()
        if not answer_info1 and not answer_info2:
            self.do_answer_client(client, server, default_answer)
        else:
            answer_strategy1 = answer_info1['answer_strategy']
            answer_strategy2 = answer_info2['answer_strategy']

            answer1 = answer_info1['replay_answer']
            answer2 = answer_info1['replay_answer']

            if answer_strategy1 == AnswerStretegy.NotFindData and answer_strategy2 == AnswerStretegy.NotFindData:
                log.info("[nlg]发现所有回答策略均为数据库未知")
                self.do_answer_client(client, server, answer1)
            elif answer_strategy1 == AnswerStretegy.FindSuccess:
                log.info("[nlg]-answer_strategy1 is success")
                final_answer = answer1
                if answer_strategy2 == AnswerStretegy.FindSuccess:
                    final_answer = final_answer + "\n" + answer2
                self.do_answer_client(client, server, final_answer)

            elif answer_strategy2 == AnswerStretegy.FindSuccess:
                log.info("[nlg]-answer_strategy2 is success")
                final_answer = answer2
                if answer_strategy1 == AnswerStretegy.FindSuccess:
                    final_answer = final_answer + "\n" + answer1
                self.do_answer_client(client, server, final_answer)
            else:
                log.info("[nlg]回答策略未知，返回默认回答")
                self.do_answer_client(client, server, default_answer)
