"""
Created by xiedong
@Date: 2023/8/8 9:09
"""
from py2neo import Graph
from combine_sp_ie.config.logger_conf import my_log
from combine_sp_ie.utils.relation import translate_relation

log = my_log.logger


class Neo4jService():
    def __init__(self, args):
        self.args = args
        try:
            self.graph = Graph(host=args.graph_host,
                               http_port=args.graph_http_port,
                               user=args.graph_user,
                               password=args.graph_password)
        except Exception as e:
            log.error("初始化链接neo4j失败！将无法查询neo4j...")

    # 伪代码示例
    def retrieve_subgraphs(self, main_entity, relation):
        # 执行查询，召回与主实体和关系匹配的子图
        query = f"MATCH (subject)-[relation:{relation}]-(object) WHERE subject.name = '{main_entity}' RETURN subject, type(relation) AS relation, object"
        result = self.graph.run(query)

        # 处理查询结果，将子图以三元组形式返回
        subgraphs = []
        for record in result:
            translated_relation_cn, translated_relation_en = translate_relation(record["relation"])
            subgraph = {
                "subject": record["subject"]["name"],
                # "relationship": record["relationship"],
                "relation": translated_relation_cn,
                "object": record["object"]["name"]
            }
            subgraphs.append(subgraph)
        return subgraphs


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialog System")
    parser.add_argument("--port", type=int, default=9001, help="WebSocket server port")
    parser.add_argument("--graph_host", type=str, default="127.0.0.1", help="neo4j host")
    parser.add_argument("--graph_http_port", type=int, default=7474, help="neo4j http_port")
    parser.add_argument("--graph_user", type=str, default="neo4j", help="neo4j user")
    parser.add_argument("--graph_password", type=str, default="123456", help="neo4j password")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    service = Neo4jService(args=args)

    subgraphs = service.retrieve_subgraphs('糖尿病', 'Reason_Disease')
    print()
