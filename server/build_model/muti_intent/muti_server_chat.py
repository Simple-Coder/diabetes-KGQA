"""
Created by xiedong
@Date: 2023/5/29 14:29
"""
from websocket_server import WebsocketServer
from logger_conf import my_log

logger = my_log.logger

# 创建 WebSocket 服务器
server = WebsocketServer(host='127.0.0.1', port=8765)

# 客户端列表，用于存储所有连接的客户端和上下文状态
clients = {}


# 处理客户端连接事件
def client_connected(client, server):
    # 将新连接的客户端添加到客户端列表，并初始化上下文状态
    clients[client['id']] = {"context": None}
    logger.info("clientId:{} 连接成功".format(client['id']))


# 处理客户端断开连接事件
def client_disconnected(client, server):
    # 移除客户端和上下文状态
    clients.pop(client['id'], None)
    logger.info("clientId:{} 连接断开".format(client['id']))


# 处理客户端消息事件
def message_received(client, server, message):
    logger.info("【Server】 接收到客户端clientId:{} 发来的消息:{}".format(client['id'], message))
    # 在这里可以根据接收到的消息进行意图识别和上下文处理
    intents, context = recognize_intents(message, clients[client['id']]["context"])

    # 更新客户端的上下文状态
    clients[client['id']]["context"] = context

    # 根据意图进行相应的处理
    handled_intents = set()
    for intent in intents:
        if intent not in handled_intents:
            handle_intent(client, server, intent)
            handled_intents.add(intent)


# 处理意图
def handle_intent(client, server, intent):
    # 根据意图进行相应的处理
    if intent == "intent1":
        handle_intent1(client, server)
    elif intent == "intent2":
        handle_intent2(client, server)
    else:
        handle_default_intent(client, server)


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
    response = "无法识别意图，请重新输入"
    server.send_message(client, response)


# 模拟意图识别和上下文处理，返回识别到的意图列表和更新后的上下文
def recognize_intents(message, context):
    # 在这里进行意图识别和上下文处理的操作，返回识别到的意图列表和更新后的上下文
    # 例如，可以使用预训练的模型和规则来进行意图识别和上下文处理
    intents = ["intent1", "intent2", "intent1"]
    updated_context = context
    return intents, updated_context


# 注册事件处理函数
server.set_fn_new_client(client_connected)
server.set_fn_client_left(client_disconnected)
server.set_fn_message_received(message_received)

logger.info("[websocket 服务端启动成功!]")
# 启动 WebSocket 服务器
server.run_forever()
