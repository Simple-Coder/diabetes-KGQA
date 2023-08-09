"""
Created by xiedong
@Date: 2023/8/7 12:31
unite_sematic_parsing_information_extraction
结合语义解析、信息抽取的各自的优势
1)Query理解 - 语义解析的开始： 在 "Query理解" 步骤中，描述了对原始查询进行句法分析，以识别用户查询的主实体、业务领域和问题类型。这一步骤使用了语义解析的思想，通过句法分析来提取问题的关键信息，如主实体和领域，从而为后续处理步骤提供基础。

2)关系识别 - 结合依存分析和信息抽取： 在 "关系识别" 步骤中，描述了利用依存分析强化问题主干，并通过匹配排序召回与旅游领域相关的关系。这一步骤将语义解析的方法（依存分析）与信息抽取的方法（关系召回与排序）结合在一起，通过分析问题的依存关系来加强问题主干，然后识别出与问题相关的关系。

3)子图召回 - 结合主实体和关系： 在 "子图召回" 步骤中，描述了根据解析的主实体和关系，召回图谱中符合条件的子图。这一步骤将语义解析（主实体和关系）与信息抽取（子图召回）相结合，通过识别主实体和关系来构建与查询相关的子图。

综合这些步骤，语义解析和信息抽取的方法在不同阶段相互结合，以实现整个流程的目标：从用户查询中理解问题、识别关系、召回子图并最终排序答案。这种结合的方式旨在充分利用两种方法的优势，以提高问答系统在知识图谱中的性能。
"""
# main.py

from combine_sp_ie.nlu.nlu_processor import NLU
from combine_sp_ie.nlg.nlg_processor import NLG
from combine_sp_ie.kg.kgqa_processor import KGQAProcessor

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialog System")
    parser.add_argument("--port", type=int, default=9001, help="WebSocket server port")
    parser.add_argument("--graph_host", type=str, default="127.0.0.1", help="neo4j host")
    parser.add_argument("--graph_http_port", type=int, default=7474, help="neo4j http_port")
    parser.add_argument("--graph_user", type=str, default="neo4j", help="neo4j user")
    parser.add_argument("--graph_password", type=str, default="123456", help="neo4j password")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 初始化 NLU 和 NLG
    nlu = NLU(args)
    nlg = NLG(args)
    kgqa_processor = KGQAProcessor(args)

    # 输入查询
    query = "故宫门票价格是多少？"

    # 通过 NLU 进行查询理解和关系识别
    main_entity, recognized_relation = nlu.process_nlu(query)

    # 在这里可以根据 recognized_relation 做进一步的处理，例如从数据库中获取相关信息等
    subgraph = kgqa_processor.search_sub_graph(query, '', main_entity, recognized_relation)

    # 通过 NLG 生成回复
    response = nlg.generate_response(query, subgraph)

    # 输出生成的回复
    print(response)


if __name__ == "__main__":
    main()
