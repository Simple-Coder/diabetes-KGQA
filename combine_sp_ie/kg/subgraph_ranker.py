"""
Created by xiedong
@Date: 2023/8/8 12:49
"""
import combine_sp_ie.config.wrapper as wrapper
import torch
from combine_sp_ie.config.base_config import SubGraphConfig
from muti_server.utils.logger_conf import my_log

log = my_log.logger


class SubgraphRanker:
    def __init__(self, args):
        self.args = args
        self.subgraph_config = SubGraphConfig()

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

    def calculate_similarity(self, query_embedding, subgraph_triplet):
        node_embedding, relationship_embedding, related_node_embedding = subgraph_triplet

        # 计算加权平均嵌入
        node_weight, rel_weight, related_node_weight = 0.4, 0.3, 0.3
        subgraph_embedding = node_weight * node_embedding + rel_weight * relationship_embedding + related_node_weight * related_node_embedding

        # 计算余弦相似度
        similarity = torch.cosine_similarity(query_embedding, subgraph_embedding, dim=1)
        return similarity

    def rank_and_filter_subgraphs(self, query, constraint, subgraphs):
        if not subgraphs or len(subgraphs) == 0:
            log.warn("[答案排序] query:{}召回子图为空，排序结束".format(query))
            return None
        # 1、query_embedding
        query_embedding = self.encode_query(query)

        ranked_and_filtered_subgraphs = sorted(
            (graph for graph in subgraphs if
             self.calculate_similarity(query_embedding,
                                       graph["embedding"]) > self.subgraph_config.subgraph_recall_match_threshold),
            key=lambda x: self.calculate_similarity(query_embedding, x["embedding"]),
            reverse=True
        )

        return ranked_and_filtered_subgraphs
        # # 假设相关性分数已经计算好，可以根据实际情况替换为实际的分数计算方法
        # sorted_subgraphs = sorted(subgraphs, key=lambda x: self.calculate_subgraph_score(query, x), reverse=True)
        #
        # filtered_subgraphs = [subgraph for subgraph in sorted_subgraphs if self.meets_constraint(subgraph, constraint)]
        #
        # # 选取Top1作为答案
        # if filtered_subgraphs:
        #     return filtered_subgraphs[0]
        # else:
        #     return None

    def calculate_subgraph_score(self, query, subgraph):
        # 根据查询和子图计算相关性分数，这里是示例，您可以根据实际情况进行计算
        # score = some_scoring_function(query, subgraph)
        # return score
        pass

    def meets_constraint(self, subgraph, constraint):
        # 根据约束条件判断子图是否满足约束，这里是示例，您可以根据实际情况进行判断
        pass
        # return some_constraint_check(subgraph, constraint)
