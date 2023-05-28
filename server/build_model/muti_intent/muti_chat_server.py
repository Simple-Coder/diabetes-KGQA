"""
Created by xiedong
@Date: 2023/5/28 12:16
"""
import json
import random
from do_predict import MutiPredictWrapper
from muti_config import Args
from muti_chat_config import gossip_corpus
from muti_utils import init_logger, load_user_dialogue_context, dump_user_dialogue_context
from muti_chat_modules import medical_robot

args = Args()
predict_wrapper = MutiPredictWrapper(args)


def answer_user_query(username, query):
    # 意图识别与槽位识别结果
    query_result = predict_wrapper.predict(query)
    # 获取意图列表
    query_intents = sorted(query_result[0], key=lambda x: x[1], reverse=True)
    # 获取槽位列表
    query_slots = query_result[1]

    # 获取最强烈的意图
    user_intent = query_intents[0][0] if query_intents else "others"
    user_intent_intensity = query_intents[0][1] if query_intents else 0

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
                              query_slots=query_slots)
        if 'visison_data' in reply:
            visison_data = reply.get("visison_data")
        if reply["slot_values"]:
            dump_user_dialogue_context(username, reply)
        reply = reply.get("replay_answer")
        return reply


def gossip_robot(intent):
    return random.choice(gossip_corpus.get(intent))


from websocket_server import WebsocketServer


# 客户端连接时的回调函数
def new_client(client, server):
    print("【服务端】：建立连接成功，客户端id %d" % client['id'])
    # server.send_message_to_all("A new client has joined!")


# 接收到客户端消息时的回调函数
def message_received(client, server, message):
    print("Client(%d) 发来消息: %s" % (client['id'], message))
    # 回答问句
    queryJson = json.loads(message)
    query_username = queryJson['username']
    query_text = queryJson['query']
    answer = answer_user_query(query_username, query_text)
    server.send_message_to_all("Client(%d) said: %s" % (client['id'], json.dumps(answer)))


if __name__ == '__main__':
    # init log
    init_logger()

    # start sever
    # 创建WebSocket服务器实例，监听在指定端口
    server = WebsocketServer('0.0.0.0', 9001)
    # 设置回调函数
    server.set_fn_new_client(new_client)
    server.set_fn_message_received(message_received)
    print('服务端启动成功')
    # 启动服务器
    server.run_forever()
