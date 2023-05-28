"""
Created by xiedong
@Date: 2023/5/28 12:37
"""

import os
import logging
import json

LOGS_DIR = "./logs"


def setup_logger():
    # 配置日志输出
    logging.basicConfig(level=logging.INFO)

    # 创建日志记录器
    logger = logging.getLogger(__name__)

    # 定义日志消息格式
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # 配置日志格式
    formatter = logging.Formatter(log_format)

    # 创建日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 将日志处理器添加到记录器
    logger.addHandler(console_handler)

    return logger

    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO)


def dump_user_dialogue_context(user, data):
    path = os.path.join(LOGS_DIR, '{}.json'.format(str(user)))
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4,
                           separators=(', ', ': '), ensure_ascii=False))


def load_user_dialogue_context(user):
    path = os.path.join(LOGS_DIR, '{}.json'.format(str(user)))
    if not os.path.exists(path):
        return {"choice_answer": "hi，机器人小谢很高心为您服务", "slot_values": None}
    else:
        with open(path, 'r', encoding='utf8') as f:
            data = f.read()
            return json.loads(data)
