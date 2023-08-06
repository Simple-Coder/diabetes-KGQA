"""
Created by xiedong
@Date: 2023/8/5 21:33
"""


class SubGraphConfig:
    subgraph_recall_size_limit = 100
    subgraph_recall_match_threshold = 0.5


class SystemConfig:
    # 默认使用语义槽回答、白名单使用子图召回方式回答
    sub_graph_white_users = ['2']
