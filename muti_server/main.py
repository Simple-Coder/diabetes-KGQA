"""
Created by xiedong
@Date: 2023/6/5 15:25
"""
import argparse
from muti_server.robot_server.robot_server import RobotWebsocketServer


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
    server.start()
