"""
Created by xiedong
@Date: 2023/5/30 13:03
"""
import json
import random
from logger_conf import my_log
from muti_chat_config import gossip_corpus, semantic_slot
from logger_conf import my_log
from muti_chat_modules import gossip_robot, medical_robot
from muti_config import Args
from cust_exception import NoIntentsException

args = Args()
logger = my_log.logger


def handle_all_intents(client, server, intent1, intent2, context):
    pass


def answer_user_query(username, query, user_intent, user_intent_intensity, all_slots):
    """
    返回问题答案
    :param username: 提问人username
    :param query: 原始提问局
    :param user_intent: 问句意图
    :param user_intent_intensity: 意图强度
    :param all_slots: 识别的槽位
    :return: 回答
    """
    # 闲聊意图
    if user_intent in ["greet", "goodbye", "deny", "isbot"]:
        answer = gossip_robot(user_intent)
        return answer
    # 接受意图
    elif user_intent == "accept":
        answer_context = load_user_dialogue_context(username)
        answer = answer_context.get("choice_answer")

        return answer
    # 诊断意图
    else:
        reply = medical_robot(user_name=username,
                              query=query,
                              query_intent=user_intent,
                              query_intensity=user_intent_intensity,
                              query_slots=all_slots)
        reply["query"] = query
        # 上下文记录
        dump_user_dialogue_context(username, reply)

        return reply.get("replay_answer")
        # if 'visison_data' in reply:
        #     visison_data = reply.get("visison_data")
        # if reply["slot_values"]:
        #     dump_user_dialogue_context(username, reply)
        #     reply = reply.get("replay_answer")
        #     return user_intent, reply
        # else:
        #     return "deny", reply.get("replay_answer")
