"""
Created by xiedong
@Date: 2023/5/30 12:51
"""
import json
import random
from do_predict import MutiPredictWrapper
from logger_conf import my_log
from muti_chat_config import gossip_corpus, semantic_slot
from muti_config import Args
from cust_exception import NoIntentsException
from answer_handler import answer_user_query

args = Args()
predict_wrapper = MutiPredictWrapper(args)
logger = my_log.logger


# 模拟意图识别和上下文处理，返回识别到的意图列表和更新后的上下文
def recognize_intents(message, context):
    # 在这里进行意图识别和上下文处理的操作，返回识别到的意图列表和更新后的上下文
    # 例如，可以使用预训练的模型和规则来进行意图识别和上下文处理

    logger.info("【Server】recognize_intents---正在识别 username:【{}】输入的query:【{}】".format(context.getusername(),
                                                                                            context.getquery()))
    predict_result = predict_wrapper.predict(message)

    # 处理结果
    all_intents = predict_result[0][:args.muti_intent_threshold_num]
    all_slots = predict_result[1]
    logger.info("【Server】recognize_intents---识别结束,username:【{}】输入的query:【{}】,识别结果意图集合:【{}】,槽位结果:【{}】"
                .format(context.getusername(), context.getquery(), ''.join(map(str, all_intents)),
                        ''.join(map(str, all_slots))))
    if not all_intents:
        raise NoIntentsException(semantic_slot['others'])

    context.setAllSlots(all_slots)

    return all_intents, context


# 处理意图
def handle_intent(client, server, intent):
    # 根据意图进行相应的处理
    if intent in gossip_corpus.keys():
        handle_gossip_rot(client, server, intent)
    elif intent == "intent1":
        handle_intent1(client, server)
    elif intent == "intent2":
        handle_intent2(client, server)
    else:
        handle_default_intent(client, server)


# 处理闲聊意图
def handle_gossip_rot(client, server, intent):
    # 在这里可以编写具体的处理逻辑
    answer = random.choice(gossip_corpus.get(intent))
    server.send_message(client, str(answer))
    logger.info("【Server】intent：【{}】闲聊意图已回答：【{}】", intent, answer)


# 处理意图1
def handle_intent1(client, server):
    # 在这里可以编写具体的处理逻辑
    response = "处理意图1的回复"
    server.send_message(client, response)


# 处理意图2
def handle_intent2(client, server):
    # 在这里可以编写具体的处理逻辑
    response = "处理意图2的回复"
    server.send_message(client, response)


# 处理默认意图
def handle_default_intent(client, server):
    # 在这里可以编写默认意图的处理逻辑
    response = "服务异常啦~请稍后重试"
    server.send_message(client, response)


# 处理默认意图
def handle_default_intent(client, server, answer):
    # 在这里可以编写默认意图的处理逻辑
    server.send_message(client, answer)
