"""
Created by xiedong
@Date: 2023/8/8 12:49
"""
from combine_sp_ie.kg.neo4j_client import Neo4jClient
from combine_sp_ie.utils.relation import translate_relation


class SubgraphRetriever:
    def __init__(self, args):
        self.args = args
        # 初始化连接Neo4j数据库的操作，具体实现需要根据您的环境和需求来完成
        self.neo4j_client = Neo4jClient(args)

    def retrieve_subgraphs(self, main_entity, relation):
        # 执行查询，召回与主实体和关系匹配的子图
        query = f"MATCH (subject)-[relation:{relation}]-(object) WHERE subject.name = '{main_entity}' RETURN subject, type(relation) AS relation, object"
        result = self.neo4j_client.execute(query)

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
