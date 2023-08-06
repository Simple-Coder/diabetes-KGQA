"""
Created by xiedong
@Date: 2023/6/6 21:52
"""
import json
import time

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
        self.do_answer_client_txt(client, server, answer)

    def get_default_answer(self):
        return semantic_slot.get("others")["replay_answer"]

    def do_answer_client_txt(self, client, server, answer):
        try:
            result = {'answer_type': 1, 'answer': answer}
            server.send_message(client, json.dumps(result, ensure_ascii=False))
        except Exception as e:
            log.error("[nlg] do_answer_client error:{}".format(e))

    def do_answer_client_json(self, client, server, answer):
        try:
            result = {'answer_type': 2, 'answer': answer}
            server.send_message(client, json.dumps(result, ensure_ascii=False))
        except Exception as e:
            log.error("[nlg] do_answer_client error:{}".format(e))

    def handle_medical(self, client, server, answer_info1):
        if not answer_info1:
            self.do_answer_client_txt(client, server, self.get_default_answer())
        else:
            self.do_answer_client_txt(client, server, answer_info1.get('replay_answer'))
            # visison_data
            visison_data = answer_info1.get('visison_data')
            if visison_data:
                self.do_answer_client_json(client, server, visison_data)

    def handle_accept_intent(self, dialog_context, client, server):
        try:
            log.info("[handle_accept_intent]-start")
            history_sematics = dialog_context.get_history_semantics()
            if len(history_sematics) == 0:
                log.warn("[handle_accept_intent]-no history_sematics")
                self.do_answer_client_txt(client, server, self.get_default_answer())
            else:
                last_answer_info = self.find_last_accept_answer(history_sematics)
                if last_answer_info:
                    log.info("[handle_accept_intent]-last_answer_info find success")
                    self.do_answer_client_txt(client, server, last_answer_info.get('choice_answer'))
                else:
                    log.warn("[handle_accept_intent]-last_answer_info find fail")
                    self.do_answer_client_txt(client, server, self.get_default_answer())
        except Exception as e:
            log.error("handle_accept_intent error:{}".format(e))
            self.do_answer_client_txt(client, server, self.get_default_answer())

    def find_last_accept_answer(self, history_sematics):
        last_answer_info = None
        try:
            for sematic in history_sematics:
                infos = sematic.get_intent_infos()
                for intent in infos:
                    if intent.get_intent_enum() == IntentEnum.Clarify:
                        if intent.get_answer_info():
                            last_answer_info = intent.get_answer_info()
                            raise Exception('find answer info')
        except Exception as e:
            pass
        return last_answer_info

    def handle_all_medical(self, client, server, answer_info1, answer_info2):
        default_answer = self.get_default_answer()
        if not answer_info1 and not answer_info2:
            self.do_answer_client_txt(client, server, default_answer)
        else:
            answer_strategy1 = answer_info1['answer_strategy']
            answer_strategy2 = answer_info2['answer_strategy']

            answer1 = answer_info1['replay_answer']
            answer2 = answer_info2['replay_answer']

            if answer_strategy1 == AnswerStretegy.NotFindData and answer_strategy2 == AnswerStretegy.NotFindData:
                log.info("[nlg]发现所有回答策略均为数据库未知")
                self.do_answer_client_txt(client, server, answer1)
            elif answer_strategy1 == AnswerStretegy.FindSuccess:
                log.info("[nlg]-answer_strategy1 is success")
                final_answer = answer1
                if answer_strategy2 == AnswerStretegy.FindSuccess:
                    final_answer = final_answer + "\n" + answer2
                    log.info("[nlg]-answer_strategy2 is success")
                self.do_answer_client_txt(client, server, final_answer)

            elif answer_strategy2 == AnswerStretegy.FindSuccess:
                log.info("[nlg]-answer_strategy2 is success")
                final_answer = answer2
                if answer_strategy1 == AnswerStretegy.FindSuccess:
                    final_answer = final_answer + "\n" + answer1
                self.do_answer_client_txt(client, server, final_answer)
            else:
                log.info("[nlg]回答策略未知，返回默认回答")
                self.do_answer_client_txt(client, server, default_answer)

    def handle_medical_clarify(self, client, server, intent_info1, intent_info2):
        try:
            if not intent_info1 and not intent_info2:
                self.do_answer_client_txt(client, server, self.get_default_answer())
                return
            answer_info1 = intent_info1.get_answer_info()
            answer_info2 = intent_info2.get_answer_info()
            self.do_answer_client_txt(client, server, answer_info1.get('replay_answer'))
            # visison_data
            visison_data = answer_info1.get('visison_data')
            if visison_data:
                self.do_answer_client_json(client, server, visison_data)
            log.info("[nlg] handle_medical_clarify,medical-send success,will sleep 1s")
            time.sleep(1)
            self.do_answer_client_txt(client, server, answer_info2.get('replay_answer'))
            log.info("[nlg] handle_medical_clarify,clarify-send success")
        except Exception as e:
            log.error("[nlg] handle_medical_clarify error:{}".format(e))
            self.do_answer_client_txt(client, server, self.get_default_answer())

    def handle_others(self, client, server):
        self.do_answer_client_txt(client, server, self.get_default_answer())
        # TODO：增加动态添加知识图谱的能力

    def handle_sub_graph_answer(self, client, server, dialog_context):
        self.do_answer_sub_graph_txt(client, server, dialog_context)

    def generate_response(self, client, server, dialog_context):
        current_semantic = dialog_context.get_current_semantic()
        history_sematics = dialog_context.get_history_semantics()

        try:
            if not current_semantic:
                log.error("[nlg] 当前语义识别为null，将采用默认回答")
                self.do_answer_client_txt(client, server, self.get_default_answer())
                return

            # 获取意图
            current_intent_infos = current_semantic.get_intent_infos()

            intent_info1 = current_intent_infos[0]
            intent1 = intent_info1.get_intent()
            intent_enum1 = intent_info1.get_intent_enum()
            answer_info1 = intent_info1.get_answer_info()

            intent_info2 = current_intent_infos[1]
            intent2 = intent_info2.get_intent()
            intent_enum2 = intent_info2.get_intent_enum()
            answer_info2 = intent_info2.get_answer_info()
            log.info("[nlg] intent_enum1:{} intent_enum2:{} start".format(intent_enum1, intent_enum2))
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
                self.handle_others(client, server)
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
            # 1个闲聊，一个澄清：优先闲聊
            elif intent_enum1 == IntentEnum.Gossip and intent_enum2 == IntentEnum.Accept:
                self.handle_gossip(intent1, client, server)
                # 1个诊断，一个闲聊：只处理诊断
            elif intent_enum1 == IntentEnum.Medical and intent_enum2 == IntentEnum.Gossip:
                self.handle_medical(client, server, answer_info1)
                # 1个诊断，一个澄清：先诊断，再澄清
            elif intent_enum1 == IntentEnum.Medical and intent_enum2 == IntentEnum.Clarify:
                self.handle_medical_clarify(client, server, intent_info1, intent_info2)
                # 1个诊断，一个未知：只诊断
            elif intent_enum1 == IntentEnum.Medical and intent_enum2 == IntentEnum.Others:
                self.handle_medical(client, server, answer_info1)
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
                self.handle_others(client, server)
                # 一个未知，一个澄清：学习中
            elif intent_enum1 == IntentEnum.Others and intent_enum2 == IntentEnum.Clarify:
                self.handle_others(client, server)
                # 一个未知，一个诊断：学习中
            elif intent_enum1 == IntentEnum.Others and intent_enum2 == IntentEnum.Medical:
                self.handle_others(client, server)
                # 1为接受，找上下文回答
            elif intent_enum1 == IntentEnum.Accept:
                self.handle_accept_intent(dialog_context, client, server)
                # 2为接受，找上下文回答
            elif intent_enum2 == IntentEnum.Clarify:
                self.handle_gossip(intent1, client, server)
            else:
                log.info("[nlg]未识别到回答策略，intent1:{},intent2:{}".format(json_str(intent_info1),
                                                                              json_str(intent_info2)))
        except Exception as e:
            log.error("nlg 生成回答异常:{}".format(e))
            # server.send_message(client, 'NLG模块异常啦~~')
            server.send_message(client, self.get_default_answer())
        finally:
            dialog_context.add_history_semantic(current_semantic)

    def do_answer_sub_graph_txt(self, client, server, dialog_context):
        try:
            # 1、获取语义信息
            current_semantic = dialog_context.get_current_semantic()
            # 2、获取子图信息
            sub_graphs_answers = current_semantic.get_answer_sub_graphs()
            if not sub_graphs_answers or len(sub_graphs_answers) == 0:
                log.error("[nlg] sub_graphs_answer is none")
                server.send_message(client, self.get_default_answer())
                return

            # 3、打印回答
            for answer in sub_graphs_answers:
                print(f"{answer['node']} - {answer['relationship']} - {answer['related_node']}")

            # 4、合并回答
            from collections import defaultdict
            merged_answers = defaultdict(list)
            for answer in sub_graphs_answers:
                key = (answer['node'], answer['relationship'])
                merged_answers[key].append(answer['related_node'])

            # 根据合并后的数据生成回答字符串
            formatted_answers = []
            for key, values in merged_answers.items():
                node, relationship = key
                related_nodes = '、 '.join(values)
                formatted_answers.append(f"'{node}'的'{relationship}'如下：{related_nodes}")

            # 打印合并后的回答
            for answer in formatted_answers:
                print(answer)
                self.do_answer_client_txt(client, server, answer)
        except Exception as e:
            log.error("[nlg] recall subgraph answer error {}".format(e))
            server.send_message(client, self.get_default_answer())
