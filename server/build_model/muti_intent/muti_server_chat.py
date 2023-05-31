"""
Created by xiedong
@Date: 2023/5/29 14:29
"""

from websocket_server import WebsocketServer

from cust_exception import NoIntentsException
from intent_handler import recognize_intents, handle_default_intent
from logger_conf import my_log
from muti_config import Args
from answer_handler import handle_all_intents
from user_context import UserContext
import json

args = Args()
logger = my_log.logger

# 创建 WebSocket 服务器
server = WebsocketServer(host='127.0.0.1', port=8765)

# 客户端列表，用于存储所有连接的客户端和上下文状态
clients = {}


# 处理客户端连接事件
def client_connected(client, server):
    # 将新连接的客户端添加到客户端列表，并初始化上下文状态
    clients[client['id']] = {"context": None}
    logger.info("【Server】clientId:{} 连接成功".format(client['id']))


# 处理客户端断开连接事件
def client_disconnected(client, server):
    # 移除客户端和上下文状态
    clients.pop(client['id'], None)
    logger.info("【Server】clientId:{} 连接断开".format(client['id']))


# 处理客户端消息事件
def message_received(client, server, message):
    try:
        logger.info("【Server】 接收到客户端clientId:{} 发来的消息:{}".format(client['id'], message))

        # 在这里可以根据接收到的消息进行意图识别和上下文处理

        queryJson = json.loads(message)
        query_username = queryJson['username']
        query_text = queryJson['query']
        user_context = UserContext(client['id'], query_username, query_text)
        clients[client['id']]["context"] = user_context

        # 意图识别
        recognize_intents(message, user_context)

        # 处理意图s
        handle_all_intents(client, server, user_context)

    except NoIntentsException as noIntents:
        logger.error("【Server】 识别到意图")
        handle_default_intent(client, server, noIntents.message)
    except Exception as e:
        logger.error("【Server】 异常了:【{}】".format(e))
        handle_default_intent(client, server)


if __name__ == '__main__':
    # 注册事件处理函数
    server.set_fn_new_client(client_connected)
    server.set_fn_client_left(client_disconnected)
    server.set_fn_message_received(message_received)

    logger.info("【Server】[websocket 服务端启动成功!]")
    # 启动 WebSocket 服务器
    server.run_forever()
