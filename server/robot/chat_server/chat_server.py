"""
Created by xiedong
@Date: 2023/6/5 15:12
"""
from websocket_server import WebsocketServer
from ..dialog_manager.dialog_manager import DialogManager


class WebSocketHandler:
    def __init__(self):
        self.dialog_manager = DialogManager()

    def new_client(self, client, server):
        print("New client connected")

    def client_left(self, client, server):
        print("Client disconnected")

    def message_received(self, client, server, message):
        print("Received message from client:", message)

        # 处理用户输入消息并发送响应
        response = self.dialog_manager.process_user_input(message)
        server.send_message(client, response)


class WebsocketServer:
    def __init__(self, port):
        self.server = None
        self.port = port

    def start(self):
        # 创建 WebSocket 服务器实例，并设置事件处理函数
        self.server = WebsocketServer(port=self.port)
        handler = WebSocketHandler()
        self.server.set_fn_new_client(handler.new_client)
        self.server.set_fn_client_left(handler.client_left)
        self.server.set_fn_message_received(handler.message_received)

        # 启动 WebSocket 服务器
        self.server.run_forever()
