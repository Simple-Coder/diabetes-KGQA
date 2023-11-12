"""
Created by xiedong
@Date: 2023/8/5 21:33
"""
from enum import Enum


class SubGraphConfig:
    subgraph_recall_size_limit = 100
    subgraph_recall_match_threshold = 0.5


class SystemConfig:
    # 默认使用语义槽回答、白名单使用子图召回方式回答
    sub_graph_white_users = ['2']


class MsgType:
    Login_Up = 1
    Login_Down = 2
    GetAllUserIds_Up = 3
    GetAllUserIds_Down = 4
    ASK_Up = 99
    ASK_Down = 98
