"""
Created by xiedong
@Date: 2023/6/5 15:10
"""
import argparse
from chat_server.chat_server import WebsocketServer
from utils.logger_conf import my_log

log = my_log.logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialog System")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket server port")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析启动参数
    args = parse_arguments()
    port = args.port

    # 创建 WebSocket 服务器
    websocket_server = WebsocketServer(port)
    log.info("server start success!,port:{}".format(port))
    websocket_server.start()
