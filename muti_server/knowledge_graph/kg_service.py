"""
Created by xiedong
@Date: 2023/8/4 15:44
"""
from py2neo import Graph
from muti_server.nlg.nlg_config import IntentEnum, CATEGORY_INDEX, AnswerStretegy
from muti_server.utils.logger_conf import my_log
from muti_server.nlu.nlu_utils import recognize_medical
from muti_server.utils.relation import translate_relation
import torch
import muti_server.base.wrapper as wrapper
from muti_server.base.base_config import SubGraphConfig

log = my_log.logger


class KgService(object):
    def __init__(self, args):
        self.args = args
        self.subgraph_config = SubGraphConfig()
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
                translated_relation_cn, translated_relation_en = translate_relation(record["relationship"])
                subgraph = {
                    "node": record["n"]["name"],
                    # "relationship": record["relationship"],
                    "relationship": translated_relation_cn,
                    "related_node": record["m"]["name"]
                }
                subgraphs.append(subgraph)
        return subgraphs

    def encode_query(self, query):
        """
        对用户query进行编码，后续计算相似度时使用
        :param query:
        :return:
        """
        inputs = wrapper.tokenizer.encode_plus(query, add_special_tokens=True, return_tensors='pt')
        outputs = wrapper.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)  # 平均池化操作
        return query_embedding

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

    def entity_link(self, dialog_context):
        """
        TODO：实体连接
        :param dialog_context:
        :return:
        """
        return dialog_context

    def subgraph_mapping(self, subgraphs, upLimit):
        # 遍历subgraphs，生成子图embedding和原始子图信息
        limit = 1
        subgraphs_with_embedding = []
        for subgraph in subgraphs:
            limit += 1
            if limit == upLimit:
                break
            subgraph_embedding = self.encode_subgraph(subgraph)
            subgraphs_with_embedding.append({"embedding": subgraph_embedding, "info": subgraph})

        return subgraphs_with_embedding

    def rank_answers(self, query_embedding, subgraphs):
        # ranked_subgraphs = sorted(subgraphs, key=lambda x: self.calculate_similarity(query_embedding, x["embedding"]),
        #                           reverse=True)
        ranked_and_filtered_subgraphs = sorted(
            (graph for graph in subgraphs if
             self.calculate_similarity(query_embedding,
                                       graph["embedding"]) > self.subgraph_config.subgraph_recall_match_threshold),
            key=lambda x: self.calculate_similarity(query_embedding, x["embedding"]),
            reverse=True
        )

        return ranked_and_filtered_subgraphs

    def calculate_similarity(self, query_embedding, subgraph_triplet):
        node_embedding, relationship_embedding, related_node_embedding = subgraph_triplet

        # 计算加权平均嵌入
        node_weight, rel_weight, related_node_weight = 0.4, 0.3, 0.3
        subgraph_embedding = node_weight * node_embedding + rel_weight * relationship_embedding + related_node_weight * related_node_embedding

        # 计算余弦相似度
        similarity = torch.cosine_similarity(query_embedding, subgraph_embedding, dim=1)
        return similarity

    def traslate_subgraphs(self, ranked_subgraphs):
        if not ranked_subgraphs or len(ranked_subgraphs) == 0:
            log.info("traslate_subgraphs not find ranked_subgraphs ")
            return []
        # 提取排序后的子图的原始信息，即node，relationship和related_node
        sorted_subgraphs_info = []
        for ranked_subgraph in ranked_subgraphs:
            subgraph_info = ranked_subgraph["info"]
            sorted_subgraphs_info.append(subgraph_info)
        return sorted_subgraphs_info

    def convert_relations(self, current_semantic):
        current_intent_infos = current_semantic.get_intent_infos()
        # 使用 lambda 表达式将 intent1 和 intent2 合并为列表
        combine_intents = lambda intent_info: [intent_info.get_intent(), intent_info.get_intent_enum()]
        intent_list = list(map(combine_intents, current_intent_infos))
        relations = intent_list
        return relations

    def search(self, dialog_context):
        """

        :param dialog_context:
        :return:
        """
        # 1、query embedding
        current_semantic = dialog_context.get_current_semantic()
        query = current_semantic.get_query()
        query_embedding = self.encode_query(query)

        # 2、子图召回
        entities = current_semantic.get_entities()
        relations = self.convert_relations(current_semantic)
        subgraphs = self.retrieve_subgraphs(entities, relations)

        # 3、子图embedding 与子图映射
        subgraphs_with_embedding = self.subgraph_mapping(subgraphs, self.subgraph_config.subgraph_recall_size_limit)

        # 4、答案排序 && 保留大于阈值的
        ranked_subgraphs = self.rank_answers(query_embedding, subgraphs_with_embedding)
        # 5、翻译成子图
        traslate_subgraphs = self.traslate_subgraphs(ranked_subgraphs)

        # 6、子团填充Context，后续nlg生成回复
        current_semantic.set_answer_sub_graphs(traslate_subgraphs)
        dialog_context.set_current_semantic(current_semantic)
