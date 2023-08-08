"""
Created by xiedong
@Date: 2023/8/8 9:09
"""
from py2neo import Graph

from combine_sp_ie.config.logger_conf import my_log

log = my_log.logger


class Neo4jClient():
    def __init__(self, args):
        self.args = args
        try:
            self.graph = Graph(host=args.graph_host,
                               http_port=args.graph_http_port,
                               user=args.graph_user,
                               password=args.graph_password)
        except Exception as e:
            log.error("初始化链接neo4j失败！将无法查询neo4j...")

    def execute(self, cql):
        result = self.graph.run(cql)
        return result
