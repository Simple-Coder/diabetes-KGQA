# """
# Created by xiedong
# @Date: 2023/5/28 12:16
# """
# import json
# import logging
# import random
# import time
#
# from do_predict import MutiPredictWrapper
# from muti_config import Args
# from muti_chat_config import gossip_corpus, semantic_slot
# from muti_utils import load_user_dialogue_context, dump_user_dialogue_context
# from muti_chat_modules import medical_robot
# from logger_conf import my_log
#
# args = Args()
# predict_wrapper = MutiPredictWrapper(args)
# logger = my_log.logger
#
#
# # myLog = MyLog()
#
#
# # 获取日志记录器
# # logger = setup_logger()
#
# def answer_user_query(username, query, user_intent, user_intent_intensity,
#                       all_slots, all_intents_infos):
#     """
#     返回问题答案
#     :param username: 提问人username
#     :param query: 原始提问局
#     :param user_intent: 问句意图
#     :param user_intent_intensity: 意图强度
#     :param all_slots: 识别的槽位
#     :return: 回答
#     """
#     # 闲聊意图
#     if user_intent in ["greet", "goodbye", "deny", "isbot"]:
#         answer = gossip_robot(user_intent)
#         return answer
#     # 接受意图
#     elif user_intent == "accept":
#         answer_context = load_user_dialogue_context(username)
#         answer = answer_context.get("choice_answer")
#
#         return answer
#     # 诊断意图
#     else:
#         reply = medical_robot(user_name=username,
#                               query=query,
#                               query_intent=user_intent,
#                               query_intensity=user_intent_intensity,
#                               query_slots=all_slots)
#         reply["query"] = query
#         reply["all_intents_infos"] = all_intents_infos
#
#         # 上下文记录
#         dump_user_dialogue_context(username, reply)
#
#         return reply.get("replay_answer")
#         # if 'visison_data' in reply:
#         #     visison_data = reply.get("visison_data")
#         # if reply["slot_values"]:
#         #     dump_user_dialogue_context(username, reply)
#         #     reply = reply.get("replay_answer")
#         #     return user_intent, reply
#         # else:
#         #     return "deny", reply.get("replay_answer")
#
#
# def gossip_robot(intent):
#     return random.choice(gossip_corpus.get(intent))
#
#
# from websocket_server import WebsocketServer
#
#
# # 客户端连接时的回调函数
# def new_client(client, server):
#     print("【服务端】：建立连接成功，客户端id %d" % client['id'])
#     # server.send_message_to_all("A new client has joined!")
#
#
# # 接收到客户端消息时的回调函数
# def message_received(client, server, message):
#     print("Client(%d) 发来消息: %s" % (client['id'], message))
#     try:
#         # 回答问句
#         queryJson = json.loads(message)
#         query_username = queryJson['username']
#         query_text = queryJson['query']
#
#         # 意图识别与槽位识别结果
#         query_result = predict_wrapper.predict(query_text)
#         # 获取意图列表
#         all_intents = sorted(query_result[0], key=lambda x: x[1], reverse=True)
#         # 获取槽位列表
#         all_slots = query_result[1]
#
#         if not all_intents:
#             logger.info("未识别到用户输入【{}】对应的意图".format(query_text))
#             server.send_message_to_all(str(semantic_slot["others"].get("replay_answer")))
#         else:
#
#             # 获取最强烈的意图
#             # user_intent = all_intents[0][0] if all_intents else "others"
#             # user_intent_intensity = all_intents[0][1] if all_intents else 0
#
#             # 问答上下文
#             answer_context = load_user_dialogue_context(query_username)
#
#             all_answers = []
#             for intentAgg in all_intents[:args.muti_intent_threshold_num]:
#                 user_intent = intentAgg[0]
#                 user_intent_intensity = intentAgg[1]
#                 logger.info("开始处理intent:{} 意图强度:{}".format(user_intent, user_intent_intensity))
#                 answer = answer_user_query(query_username,
#                                            query_text,
#                                            user_intent,
#                                            user_intent_intensity,
#                                            all_slots,
#                                            all_intents)
#
#                 all_answers.append(answer)
#                 logger.info("结束处理intent:{} 意图强度:{} 处理结果: {}".format(
#                     user_intent, user_intent_intensity, answer))
#
#             # 所有答案都出问题了
#             if not all_answers:
#                 logger.info("未找到【{}】对应的回答".format(query_text))
#                 server.send_message_to_all(str(semantic_slot["others"].get("replay_answer")))
#                 return
#             # 只识别到一个意图
#             if len(all_answers) == 1:
#                 one_answer = all_answers[0]
#                 server.send_message_to_all(str(one_answer))
#                 return
#             # 识别到2个意图
#             if len(all_answers) == 2:
#                 # 意图最强的
#                 first_answer = all_answers[0]
#                 # 意图次强的
#                 second_answer = all_answers[1]
#
#                 if first_answer == "":  # 闲聊、拒绝直接返回；不再处理第二个意图
#                     server.send_message_to_all(str(''))
#                     return
#
#                 if first_answer == "":  # 澄清则看第二个意图
#                     pass
#
#                 if first_answer == "":  #accept则直接返回，等待处理第二个意图
#                     pass
#
#         # TODO:待发送
#         # server.send_message_to_all(str(answer))
#     except Exception as r:
#         server.send_message_to_all(str('服务器发生异常啦~~~'))
#         logger.error("未知错误 %s " % r)
#         # data = {}
#         # data["data"] = '服务器发生异常啦~~~'
#         # data["code"] = 20000
#         # data["message"] = 'success'
#         # server.send_message_to_all(str(data))
#
#
# if __name__ == '__main__':
#     # start sever
#     # 创建WebSocket服务器实例，监听在指定端口
#     server = WebsocketServer('0.0.0.0', 9001)
#     # 设置回调函数
#     server.set_fn_new_client(new_client)
#     server.set_fn_message_received(message_received)
#     print('服务端启动成功')
#     # 启动服务器
#     server.run_forever()
