"""
Created by xiedong
@Date: 2023/6/5 15:25
"""
import argparse
from muti_server.web_socket_server.robot_server import RobotWebsocketServer
from muti_server.utils.logger_conf import my_log

log = my_log.logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialog System")
    parser.add_argument("--port", type=int, default=9001, help="WebSocket server port")
    parser.add_argument("--graph_host", type=str, default="127.0.0.1", help="neo4j host")
    parser.add_argument("--graph_http_port", type=int, default=7474, help="neo4j http_port")
    parser.add_argument("--graph_user", type=str, default="neo4j", help="neo4j user")
    parser.add_argument("--graph_password", type=str, default="123456", help="neo4j password")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析启动参数
    args = parse_arguments()

    # 创建 WebSocket 服务器
    server = RobotWebsocketServer(args)
    log.info("服务端启动成功，端口:{}".format(args.port))
    server.start()
