"""
Created by xiedong
@Date: 2023/6/5 15:25
"""
import argparse
from muti_server.robot_server.robot_server import RobotWebsocketServer
from muti_server.utils.logger_conf import my_log

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
    server = RobotWebsocketServer(port)
    log.info("服务端启动成功，端口:{}".format(port))
    server.start()
