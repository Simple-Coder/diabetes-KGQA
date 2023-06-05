"""
Created by xiedong
@Date: 2023/6/5 12:37
启动类
"""
import argparse
from dialog_manager import DialogManager
from websocket_server import WebsocketServer

from dialog_manager import DialogManager
from websocket_server import WebsocketServer


def new_client(client, server):
    # 处理新客户端连接的逻辑
    pass


def message_received(client, server, message):
    system_response = dialog_manager.process_user_input(message)
    server.send_message(client, system_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Dialog System")
    parser.add_argument("-p", "--port", type=int, default=8765, help="WebSocket server port")
    args = parser.parse_args()

    dialog_manager = DialogManager()

    server = WebsocketServer(args.port, host='localhost')
    server.set_fn_new_client(new_client)
    server.set_fn_message_received(message_received)

    server.run_forever()
