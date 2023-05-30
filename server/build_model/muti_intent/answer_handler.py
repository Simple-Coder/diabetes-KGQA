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
from user_context import UserContext

args = Args()
logger = my_log.logger


def handle_all_intents(client, server, intents, user_context):
    username = user_context.getusername()
    all_slots = user_context.getAllSlots()

    intent_info1 = intents[0]
    intent1 = intent_info1[0]
    intent1_intensity = intent_info1[1]
    answer1 = answer_user_query(username, intent1, intent1_intensity, all_slots, user_context)
    logger.info("准备回答：【{}】".format(answer1))
    server.send_message(client, answer1)


def answer_user_query(username, user_intent, user_intent_intensity, all_slots, user_context):
    logger.info(
        "answer_user_query,username:【{}】input:【{}】,user_intent:【{}】,intensity:【{}】".format(username, user_intent,
                                                                                           user_context.getquery(),
                                                                                           user_intent_intensity))
    """
    返回问题答案
    :param username: 提问人username
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
        context_slot = user_context.getContextSlot()
        if context_slot:
            answer = user_context['content'].get("choice_answer")
            return answer
        else:
            logger.info("username:【{}】输入的:【{}】对应意图：【{}】,上下文不存在，正在学习中...".format(username,
                                                                                                  user_context.getquery(),
                                                                                                  user_intent))
            answer = semantic_slot.get("others").get('replay_answer')
            return answer
    # 诊断意图
    else:
        reply = medical_robot(user_name=username,
                              query_intent=user_intent,
                              query_intensity=user_intent_intensity,
                              query_slots=all_slots)
        user_context['content'] = reply
        answer = reply.get("replay_answer")
        return answer
