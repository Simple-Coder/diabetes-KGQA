"""
Created by xiedong
@Date: 2023/8/8 12:49
"""
from combine_sp_ie.config.chat_config import IntentEnum, CATEGORY_INDEX, AnswerStretegy
from combine_sp_ie.config.logger_conf import my_log
from combine_sp_ie.kg.neo4j_client import Neo4jClient
from combine_sp_ie.kg.subgraph_ranker import SubgraphRanker
from combine_sp_ie.kg.subgraph_retriever import SubgraphRetriever
from combine_sp_ie.nlu.nlu_utils import recognize_medical

log = my_log.logger


class KGQAProcessor():
    def __init__(self):
        self.subgraph_retriever = SubgraphRetriever()
        self.subgraph_ranker = SubgraphRanker()
        # 初始化连接Neo4j数据库的操作，具体实现需要根据您的环境和需求来完成
        self.neo4j_client = Neo4jClient()

    def search_sub_graph(self, query, constraint, main_entity, relation):
        # 1、子图召回
        subgraphs_with_embedding = self.subgraph_retriever.retrieve_subgrapsh_with_embedding(main_entity, relation)

        # 2、答案排序和过滤
        single_sub_graph = self.subgraph_ranker.rank_and_filter_subgraphs(query, constraint, subgraphs_with_embedding)

        return single_sub_graph
        # return ["糖尿病", "症状", ["吃的多", "喝得多", "尿量多"]]

    def search(self, slot_info, strategy):
        """
        根据语义槽获取答案回复
        :param slot_info:
        :param strategy:
        :return:
        """
        if not slot_info:
            log.error("[dst] slot_info is none,search exit")
            return None

        if not recognize_medical(strategy):
            log.error("[dst] recognize_medical false,strategy:{},not search neo4j".format(strategy))
            return None

        log.info("[dst] recognize_medical success strategy:{},will search neo4j".format(strategy))

        cql_template = slot_info.get("cql_template")
        cql_template_vision = slot_info.get("cql_template_vision")
        reply_template = slot_info.get("reply_template")
        ask_template = slot_info.get("ask_template")
        slot_values = slot_info.get("slot_values")
        # strategy = slot_info.get("intent_strategy")

        if not slot_values:
            log.error("[dst] slot_values is none,search exit")
            return slot_info

        slot_info["answer_strategy"] = AnswerStretegy.FindSuccess

        # if strategy == "accept":
        # if strategy == IntentEnum.Accept:
        if strategy == IntentEnum.Medical:
            # if strategy == IntentEnum.Medical:
            cql = []
            if isinstance(cql_template, list):
                for cqlt in cql_template:
                    cql.append(cqlt.format(**slot_values))
            else:
                cql = cql_template.format(**slot_values)

            answer = self.neo4j_searcher(cql)
            cql_vision = cql_template_vision.format(**slot_values)
            # 查询可视化vision
            visison_data = self.neo4j_searcher_vision(cql_vision)
            slot_info["visison_data"] = visison_data
            if not answer:
                slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
                slot_info["answer_strategy"] = AnswerStretegy.NotFindData
            else:
                pattern = reply_template.format(**slot_values)
                slot_info["replay_answer"] = pattern + answer

        # elif strategy == "clarify":
        elif strategy == IntentEnum.Clarify:
            # 澄清用户是否问该问题
            pattern = ask_template.format(**slot_values)
            slot_info["replay_answer"] = pattern
            # 得到肯定意图之后需要给用户回复的答案
            cql = []
            if isinstance(cql_template, list):
                for cqlt in cql_template:
                    cql.append(cqlt.format(**slot_values))
            else:
                cql = cql_template.format(**slot_values)
            answer = self.neo4j_searcher(cql)

            cql_vision = cql_template_vision.format(**slot_values)
            # 查询可视化vision
            visison_data = self.neo4j_searcher_vision(cql_vision)
            slot_info["visison_data"] = visison_data

            if not answer:
                slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
                slot_info["answer_strategy"] = AnswerStretegy.NotFindData
            else:
                pattern = reply_template.format(**slot_values)
                slot_info["choice_answer"] = pattern + answer
        # elif strategy == "deny":
        elif strategy == IntentEnum.DENY:
            slot_info["replay_answer"] = slot_info.get("deny_response")

        return slot_info

    def neo4j_searcher(self, cql_list):
        ress = ""
        if isinstance(cql_list, list):
            for cql in cql_list:
                rst = []
                data = self.neo4j_client.execute(cql).data()
                if not data:
                    continue
                for d in data:
                    d = list(d.values())
                    if isinstance(d[0], list):
                        rst.extend(d[0])
                    else:
                        rst.extend(d)

                data = "、".join([str(i) for i in rst])
                ress += data + "\n"
        else:
            data = self.neo4j_client.execute(cql_list).data()
            if not data:
                return ress
            rst = []
            for d in data:
                d = list(d.values())
                if isinstance(d[0], list):
                    rst.extend(d[0])
                else:
                    rst.extend(d)

            data = "、".join([str(i) for i in rst])
            ress += data

        return ress

    def neo4j_searcher_vision(self, cql_vision):
        try:
            data = []
            links = []
            kgdata = self.neo4j_client.execute(cql_vision).data()
            if not kgdata:
                return [data, links]
            count = 0
            for value in kgdata:
                count += 1
                relNode = value["type"]
                Relid = value["Relid"]
                pNode = value["p"]
                qNode = value["q"]
                plabel_ = value["plabel"]
                qlabel_ = value["qlabel"]
                if count == 1:
                    data.append({"id": str(qNode.identity), "name": qNode["name"], "des": qNode["name"],
                                 "category": CATEGORY_INDEX[qlabel_]})
                else:
                    data.append({"id": str(pNode.identity), "name": pNode["name"], "des": pNode["name"],
                                 "category": CATEGORY_INDEX[plabel_]})
                links.append(
                    {"source": str(qNode.identity), "target": str(pNode.identity), "value": relNode,
                     "id": str(Relid)})

            return {
                "data": data,
                "links": links
            }
        except Exception as e:
            log.error("[dst] query vision data error:{}".format(e))


if __name__ == '__main__':
    pass
