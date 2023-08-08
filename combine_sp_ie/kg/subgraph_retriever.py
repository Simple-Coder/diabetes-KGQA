"""
Created by xiedong
@Date: 2023/8/8 12:49
"""
from combine_sp_ie.kg.neo4j_client import Neo4jClient
from combine_sp_ie.utils.relation import translate_relation
from combine_sp_ie.config.base_config import SubGraphConfig
import combine_sp_ie.config.wrapper as wrapper


class SubgraphRetriever:
    def __init__(self, args):
        self.args = args
        self.subgraph_config = SubGraphConfig()
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

    def encode_subgraph(self, subgraph):
        """
        子图编码
        :param subgraph:
        :return:
        """
        node_embedding = wrapper.model(
            **wrapper.tokenizer(subgraph["node"], add_special_tokens=True, return_tensors='pt')).last_hidden_state.mean(
            dim=1)
        relationship_embedding = wrapper.model(**wrapper.tokenizer(subgraph["relationship"], add_special_tokens=True,
                                                                   return_tensors='pt')).last_hidden_state.mean(dim=1)
        related_node_embedding = wrapper.model(**wrapper.tokenizer(subgraph["related_node"], add_special_tokens=True))

        return node_embedding, relationship_embedding, related_node_embedding

    def subgraph_mapping(self, subgraphs):
        # 遍历subgraphs，生成子图embedding和原始子图信息
        subgraphs_with_embedding = []
        for subgraph in subgraphs:
            subgraph_embedding = self.encode_subgraph(subgraph)
            subgraphs_with_embedding.append({"embedding": subgraph_embedding, "info": subgraph})

        return subgraphs_with_embedding

    def retrieve_subgrapsh_with_embedding(self, main_entity, relation):
        # 1、查询原始子图
        subgraphs = self.retrieve_subgraphs(main_entity, relation)
        subgraphs = subgraphs[:self.subgraph_config.subgraph_recall_size_limit]

        # 2、子团转embedding
        subgraphs_with_embedding = self.subgraph_mapping(subgraphs)

        return subgraphs_with_embedding
