"""
Created by xiedong
@Date: 2023/4/22 21:00
"""
from py2neo import Graph
from chat_config import semantic_slot
from utils import load_user_dialogue_context

graph = Graph(host="127.0.0.1",
              http_port=7474,
              user="neo4j",
              password="123456")


def medical_robot(user_intent, user_slots, username):
    # TODO：携带用户信息
    """
    如果确定是诊断意图则使用该方法进行诊断问答
    """
    semantic_slot = semantic_parser(user_intent, user_slots, username)
    return get_answer(semantic_slot)


def neo4j_searcher(cql_list):
    ress = ""
    if isinstance(cql_list, list):
        for cql in cql_list:
            rst = []
            data = graph.run(cql).data()
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
        data = graph.run(cql_list).data()
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


def entity_link(mention, etype):
    """
    对于识别到的实体mention,如果其不是知识库中的标准称谓
    则对其进行实体链指，将其指向一个唯一实体（待实现）
    """
    return mention


def semantic_parser(user_intent, user_slots, username):
    """
    对文本进行解析
    intent = {"name":str,"confidence":float}
    """
    # intent_rst = intent_classifier(text)
    # slot_rst = slot_recognizer(text)

    if user_intent == "UNK":
        return semantic_slot.get("unrecognized")

    slot_info = semantic_slot.get(user_intent)

    # 填槽
    slots = slot_info.get("slot_list")
    slot_values = {}
    for slot in slots:
        slot_values[slot] = None
        for ent_info in user_slots:
            slot_key = ent_info[0]
            slot_val = ent_info[1]
            slot_values[slot_key] = slot_val
            # print()
            # for e in ent_info["entities"]:
            #     if slot.lower() == e['type']:
            #         slot_values[slot] = entity_link(e['word'], e['type'])

    last_slot_values = load_user_dialogue_context(username)["slot_values"]
    for k in slot_values.keys():
        if slot_values[k] is None:
            slot_values[k] = last_slot_values.get(k, None)

    slot_info["slot_values"] = slot_values

    # 根据意图强度来确认回复策略
    # conf = intent_rst.get("confidence")
    # if conf >= intent_threshold_config["accept"]:
    #     slot_info["intent_strategy"] = "accept"
    # elif conf >= intent_threshold_config["deny"]:
    #     slot_info["intent_strategy"] = "clarify"
    # else:
    #     slot_info["intent_strategy"] = "deny"
    slot_info["intent_strategy"] = "accept"
    return slot_info


def get_answer(slot_info):
    """
    根据语义槽获取答案回复
    """
    cql_template = slot_info.get("cql_template")
    reply_template = slot_info.get("reply_template")
    ask_template = slot_info.get("ask_template")
    slot_values = slot_info.get("slot_values")
    strategy = slot_info.get("intent_strategy")

    if not slot_values:
        return slot_info

    if strategy == "accept":
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
            slot_info["replay_answer"] = pattern + answer

    elif strategy == "clarify":
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
    elif strategy == "deny":
        slot_info["replay_answer"] = slot_info.get("deny_response")

    return slot_info
