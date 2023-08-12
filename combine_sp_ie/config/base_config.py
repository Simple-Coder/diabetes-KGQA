"""
Created by xiedong
@Date: 2023/8/5 21:33
"""


class SubGraphConfig:
    subgraph_recall_size_limit = 100
    subgraph_recall_match_threshold = 0.5


class GlobalConfig:
    bert_dir = 'hfl/chinese-bert-wwm-ext'
    logs_dir = '/Users/xiedong/PycharmProjects/diabetes-KGQA/combine_sp_ie/logs'
    graph_host = '127.0.0.1'
    graph_http_port = 7474
    graph_user = 'neo4j'
    graph_password = '123456'
