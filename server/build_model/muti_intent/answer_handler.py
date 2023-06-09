"""
Created by xiedong
@Date: 2023/5/30 13:03
"""
from logger_conf import my_log
from muti_chat_config import semantic_slot
from muti_chat_modules import gossip_robot, medical_robot, gossip_corpus
from muti_config import Args

args = Args()
logger = my_log.logger


def handle_all_intents(client, server, user_context):
    """
    2个均为闲聊：取第一个回答
    2个均为诊断：合并答案
    2个均为澄清：合并澄清
    2个均为未知：返回学习中

    1个闲聊，一个诊断：合并诊断
    1个闲聊，一个澄清：合并澄清
    1个闲聊，一个未知：优先闲聊

    1个诊断，一个闲聊：只处理诊断
    1个诊断，一个澄清：先诊断，再澄清
    1个诊断，一个未知：只诊断

    一个澄清，一个闲聊：只处理澄清
    一个澄清，一个诊断：合并诊断后澄清
    一个澄清，一个未知：只处理澄清

    一个未知，一个闲聊：学习中
    一个未知，一个澄清：学习中
    一个未知，一个诊断：学习中
    :param client:
    :param server:
    :param user_context:
    :return:
    """
    username = user_context.get_username()
    all_slots = user_context.get_all_slots()
    intents = user_context.get_all_intents()

    intent_info1 = intents[0]
    intent1 = intent_info1[0]
    intent1_intensity = intent_info1[1]

    intent_info2 = intents[1]
    intent2 = intent_info2[0]
    intent2_intensity = intent_info2[1]

    logger.info(
        "username:【{}】,输入：【{}】识别到意图1：【{}】，意图2：【{}】，将要检查会话策略回答".format(username,
                                                                                         user_context.get_query(),
                                                                                         intent1,
                                                                                         intent2))

    if intent1 in gossip_corpus.keys() and intent2 in gossip_corpus.keys():
        """
        2个均为闲聊：取第一个回答
        """
        logger.info("识别query：【{}】对应的2个意图均为闲聊".format(user_context.get_query()))
        answer = answer_user_query(username, intent1, intent1_intensity, all_slots, user_context)
        logger.info("准备回答：【{}】".format(answer))
        server.send_message(client, answer)

    # 2、意图1=诊断 && 意图2=诊断
    elif intent1 in semantic_slot.keys() and intent2 in semantic_slot.keys():
        """
        2个均为诊断：合并答案
        """
        logger.info("识别query：【{}】对应的2个意图均为诊断".format(user_context.get_query()))

        answer1 = answer_user_query(username, intent1, intent1_intensity, all_slots, user_context)
        answer2 = answer_user_query(username, intent2, intent2_intensity, all_slots, user_context)
        # 合并2个answer
        logger.info("准备回答：【{}】".format(answer1 + "," + answer2))
        server.send_message(client, answer1 + "," + answer2)

    # 3、意图1=意图2=其他
    elif intent1 == "others" and intent2 == "others":
        logger.info("识别query：【{}】对应的2个意图均为未知".format(user_context.get_query()))
        answer = semantic_slot.get("others").get('replay_answer')
        logger.info("准备回答：【{}】".format(answer))
        server.send_message(client, answer)
    elif intent1 in semantic_slot.keys():
        """
        1个诊断，一个闲聊：只处理诊断
        1个诊断，一个澄清：先诊断，再澄清
        1个诊断，一个未知：只诊断
        """
        if intent2 in gossip_corpus.keys() or intent2 == "others":
            logger.info(
                "识别query：【{}】,意图1:【{}】,意图2：【{}】，最强烈意图为诊断，次意图为闲聊（将丢弃）使用诊断回答".format(
                    user_context.get_query(),
                    intent1, intent2))
            answer1 = answer_user_query(username, intent1, intent1_intensity, all_slots, user_context)
            server.send_message(client, answer1)
        # if intent1 TODO:澄清

    elif intent1 == "others":
        '''
       一个未知，一个闲聊：学习中
       一个未知，一个澄清：学习中
       一个未知，一个诊断：学习中
        '''
        logger.info(
            "识别query：【{}】,意图1:【{}】,意图2：【{}】，最强烈意图为未知，采用默认回答".format(user_context.get_query(),
                                                                                        intent1, intent2))
        answer = semantic_slot.get("others").get('replay_answer')
        logger.info("准备回答：【{}】".format(answer))
        server.send_message(client, answer)
    else:
        logger.info(
            "识别query：【{}】,意图1:【{}】,意图2：【{}】未找到对应回答策略".format(user_context.get_query(), intent1, intent2))
        answer = semantic_slot.get("others").get('replay_answer')
        logger.info("准备回答：【{}】".format(answer))
        server.send_message(client, answer)


def answer_user_query(username, user_intent, user_intent_intensity, all_slots, user_context):
    logger.info(
        "answer_user_query,username:【{}】输入:【{}】,user_intent:【{}】,intensity:【{}】".format(username,
                                                                                          user_context.get_query(),
                                                                                          user_intent,
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
        context_slot = user_context.get_context_slot()
        if context_slot:
            answer = context_slot.get("choice_answer")
            return answer
        else:
            logger.info("username:【{}】输入的:【{}】对应意图：【{}】,上下文不存在，正在学习中...".format(username,
                                                                                                  user_context.get_query(),
                                                                                                  user_intent))
            answer = semantic_slot.get("others").get('replay_answer')
            return answer
    # 诊断意图
    else:
        reply = medical_robot(user_name=username,
                              query_intent=user_intent,
                              query_intensity=user_intent_intensity,
                              query_slots=all_slots)
        user_context.set_context_slot(reply)
        answer = reply.get("replay_answer")
        return answer
