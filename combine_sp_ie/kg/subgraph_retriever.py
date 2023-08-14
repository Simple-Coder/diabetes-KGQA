"""
Created by xiedong
@Date: 2023/8/8 12:49
"""
from combine_sp_ie.kg.neo4j_client import Neo4jClient
from combine_sp_ie.utils.relation import translate_relation
from combine_sp_ie.config.base_config import GlobalConfig
from combine_sp_ie.models.model_wrapper import model_service
from combine_sp_ie.config.logger_conf import my_log

log = my_log.logger


class SubgraphRetriever:
    def __init__(self):
        # 初始化连接Neo4j数据库的操作，具体实现需要根据您的环境和需求来完成
        self.neo4j_client = Neo4jClient()

    def retrieve_subgraphs(self, main_entity, relation):
        # 执行查询，召回与主实体和关系匹配的子图
        query = f"MATCH (n)-[r:{relation}]-(m) WHERE n.name CONTAINS '{main_entity}' OR m.name CONTAINS '{main_entity}' RETURN n, type(r) AS relationship, m"

        result = self.neo4j_client.execute(query)

        # 异常check
        if not result:
            log.info("[子图召回] 结果为空,main_entity:{},relation:{}".format(main_entity, relation))
            return None

        # 处理查询结果，将子图以三元组形式返回
        subgraphs = []
        for record in result:
            translated_relation_cn, translated_relation_en = translate_relation(record["relationship"])
            subgraph = {
                "node": record["n"]["name"],
                # "relationship": record["relationship"],
                "relationship": translated_relation_cn,
                "related_node": record["m"]["name"]
            }
            subgraphs.append(subgraph)
        return subgraphs

    def encode_subgraph(self, subgraph):
        """
        子图编码
        :param subgraph:
        :return:
        """
        # 使用正确的模型和分词器对象

        # 编码节点
        node_embedding = model_service.model(
            **model_service.tokenizer(subgraph["node"], add_special_tokens=True,
                                      return_tensors='pt')).last_hidden_state.mean(dim=1)

        # 编码关系
        relationship_embedding = model_service.model(
            **model_service.tokenizer(subgraph["relationship"], add_special_tokens=True,
                                      return_tensors='pt')).last_hidden_state.mean(
            dim=1)

        # 编码相关节点
        related_node_embedding = model_service.model(
            **model_service.tokenizer(subgraph["related_node"], add_special_tokens=True,
                                      return_tensors='pt')).last_hidden_state.mean(
            dim=1)

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
        if not subgraphs or len(subgraphs) == 0:
            log.warn("[子图召回embedding] 子图为空，embedding exit")
            return None

        subgraphs = subgraphs[:GlobalConfig.subgraph_recall_size_limit]

        # 2、子团转embedding
        subgraphs_with_embedding = self.subgraph_mapping(subgraphs)

        return subgraphs_with_embedding
