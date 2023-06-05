"""
Created by xiedong
@Date: 2023/6/5 15:31
"""
import json

from websocket_server import WebsocketServer
from muti_server.dialog_manager.dialog_manager import DialogManager
from muti_server.bean.user_question import QuestionInfo
from muti_server.utils.logger_conf import my_log

log = my_log.logger


class RobotWebSocketHandler:
    def __init__(self):
        self.dialog_manager = DialogManager()

    def new_client(self, client, server):
        print("New client connected")

    def client_left(self, client, server):
        print("Client disconnected")

    def message_received(self, client, server, message):
        try:
            print("Received message from client:", message)
            # 1、将用户输入封装为对象，后续使用
            question_info = self.convert_message(message)
            # 2、处理用户输入消息
            response = self.dialog_manager.process_user_input(question_info)
            # 3、写回客户端
            server.send_message(client, str(response))
        except Exception as e:
            log.error("服务器异常：{}".format(e))
            server.send_message(client, '服务器异常啦~请找小谢查看！')

    def convert_message(self, message):
        queryJson = json.loads(message)
        query_username = queryJson['username']
        query_text = queryJson['query']

        question_info = QuestionInfo()
        question_info.user_id = query_username
        question_info.user_question = query_text

        return question_info


class RobotWebsocketServer:
    def __init__(self, port):
        self.server = None
        self.port = port

    def start(self):
        # 创建 WebSocket 服务器实例，并设置事件处理函数
        self.server = WebsocketServer(port=self.port)
        handler = RobotWebSocketHandler()
        self.server.set_fn_new_client(handler.new_client)
        self.server.set_fn_client_left(handler.client_left)
        self.server.set_fn_message_received(handler.message_received)

        # 启动 WebSocket 服务器
        self.server.run_forever()


if __name__ == '__main__':
    # 创建 WebSocket 服务器
    server = RobotWebsocketServer(9999)
    print('启动test成功')
    server.start()
