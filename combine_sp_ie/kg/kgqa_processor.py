"""
Created by xiedong
@Date: 2023/8/8 12:49
"""
import argparse
from combine_sp_ie.kg.subgraph_retriever import SubgraphRetriever
from combine_sp_ie.kg.subgraph_ranker import SubgraphRanker


class KGQAProcessor():
    def __init__(self, args):
        self.args = args
        self.subgraph_retriever = SubgraphRetriever(args)
        self.subgraph_ranker = SubgraphRanker(args)

    def search_sub_graph(self, query, constraint, main_entity, relation):
        # 1、子图召回
        subgraphs_with_embedding = self.subgraph_retriever.retrieve_subgrapsh_with_embedding(main_entity, relation)

        # 2、答案排序和过滤
        single_sub_graph = self.subgraph_ranker.rank_and_filter_subgraphs(query, constraint, subgraphs_with_embedding)

        return single_sub_graph
        # return ["糖尿病", "症状", ["吃的多", "喝得多", "尿量多"]]


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
    kgqa_processor = KGQAProcessor(args)
    subgraphs = kgqa_processor.search('', '', '糖尿病', 'Reason_Disease')
    print()
