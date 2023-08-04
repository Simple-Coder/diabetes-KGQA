"""
Created by xiedong
@Date: 2023/8/4 15:44
"""
from py2neo import Graph
from muti_server.nlg.nlg_config import IntentEnum, CATEGORY_INDEX, AnswerStretegy
from muti_server.utils.logger_conf import my_log
from muti_server.nlu.nlu_utils import recognize_medical
from muti_server.utils.relation import translate_relation

log = my_log.logger


class KgService(object):
    def __init__(self, args):
        self.args = args
        try:
            self.graph = Graph(host=args.graph_host,
                               http_port=args.graph_http_port,
                               user=args.graph_user,
                               password=args.graph_password)
        except Exception as e:
            log.error("初始化链接neo4j失败！将无法查询neo4j...")


class InfoRetrieveService(KgService):
    def __init__(self, args):
        super(InfoRetrieveService, self).__init__(args)

    def retrieve_subgraphs(self, entities, relations):
        """
        检索子图
        :param entities: ["糖尿病"]
        :param relations: ["Symptom_Disease","Reason_Disease"]
        :return:
        """
        subgraphs = []
        for entity in entities:
            if not relations:
                cypher_query = f"MATCH (n)-[r]-(m) WHERE n.name CONTAINS $entity OR m.name CONTAINS $entity RETURN n, type(r) AS relationship, m"
                result = self.graph.run(cypher_query, {"entity": entity})
            else:
                relation_clause = "|".join(relations)
                cypher_query = f"MATCH (n)-[r:{relation_clause}]-(m) WHERE n.name CONTAINS $entity OR m.name CONTAINS $entity RETURN n, type(r) AS relationship, m"
                result = self.graph.run(cypher_query, {"entity": entity})

            for record in result:
                subgraph = {
                    "node": record["n"]["name"],
                    # "relationship": record["relationship"],
                    "relationship": translate_relation(record["relationship"]),
                    "related_node": record["m"]["name"]
                }
                subgraphs.append(subgraph)

