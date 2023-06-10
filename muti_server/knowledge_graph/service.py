"""
Created by xiedong
@Date: 2023/6/8 21:11
"""
from py2neo import Graph
from muti_server.nlg.nlg_config import IntentEnum
from muti_server.utils.logger_conf import my_log

log = my_log.logger


class KgService:
    def __init__(self, args):
        self.args = args
        try:
            self.graph = Graph(host=args.graph_host,
                               http_port=args.graph_http_port,
                               user=args.graph_user,
                               password=args.graph_password)
        except Exception as e:
            log.error("初始化链接neo4j失败！将无法查询neo4j...")
        # self.graph = Graph(host="127.0.0.1",
        #                    http_port=7474,
        #                    user="neo4j",
        #                    password="123456")

    def search(self, slot_info, strategy):
        """
        根据语义槽获取答案回复
        :param slot_info:
        :param strategy:
        :return:
        """
        if not slot_info:
            return None

        cql_template = slot_info.get("cql_template")
        cql_template_vision = slot_info.get("cql_template_vision")
        reply_template = slot_info.get("reply_template")
        ask_template = slot_info.get("ask_template")
        slot_values = slot_info.get("slot_values")
        # strategy = slot_info.get("intent_strategy")

        if not slot_values:
            return slot_info

        # if strategy == "accept":
        if strategy == IntentEnum.Accept:
            cql = []
            if isinstance(cql_template, list):
                for cqlt in cql_template:
                    cql.append(cqlt.format(**slot_values))
            else:
                cql = cql_template.format(**slot_values)

            cql_vision = cql_template_vision.format(**slot_values)
            answer = self.neo4j_searcher(cql)
            # 查询可视化vision
            # visison_data = neo4j_searcher_vision(cql_vision)
            # slot_info["visison_data"] = visison_data
            if not answer:
                slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
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
            answer = neo4j_searcher(cql)
            if not answer:
                slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
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
                data = self.graph.run(cql).data()
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
            data = self.graph.run(cql_list).data()
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
