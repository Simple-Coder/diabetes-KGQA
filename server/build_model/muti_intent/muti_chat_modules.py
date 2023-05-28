"""
Created by xiedong
@Date: 2023/5/28 12:25
"""
import logging
from py2neo import Graph
from muti_chat_config import semantic_slot, intent_threshold_config, CATEGORY_INDEX
from muti_utils import load_user_dialogue_context, setup_logger

logger = setup_logger()

graph = Graph(host="127.0.0.1",
              http_port=7474,
              user="neo4j",
              password="123456")


def medical_robot(user_name, query, query_intent, query_intensity, query_slots):
    """
    诊断用户
    :param user_name:用户username
    :param query: 原始问句
    :param query_intent: 问句意图
    :param query_intensity: 意图强度
    :param query_slots: 序列标注结果
    :return:
    """
    logger.info("medical_robot start username:%s query:%s query_intent:%s query_intensity:%d query_slots:%s", user_name,
                query, query_intent, query_intensity, query_slots)
    # 1、如果是其他意图则返回未知
    if query_intent == "others":
        return semantic_slot.get("others")
    # 2、获取意图对应语义槽信息
    slot_info = semantic_slot.get(query_intent)
    # 获取待填充槽列表
    slots = slot_info.get("slot_list")
    slot_values = {}
    for slot in slots:
        slot_values[slot] = None
        for ent_info in query_slots:
            slot_key = ent_info[0]
            slot_val = ent_info[1]
            slot_values[slot_key] = slot_val
            # print()
            # for e in ent_info["entities"]:
            #     if slot.lower() == e['type']:
            #         slot_values[slot] = entity_link(e['word'], e['type'])
    last_slot_values = load_user_dialogue_context(user_name)["slot_values"]
    if last_slot_values:
        for k in slot_values.keys():
            if slot_values.get(k) is None:
                slot_values[k] = last_slot_values.get(k, None)

    if None in slot_values.values():
        none_keys = [k for k, v in slot_values.items() if v is None]
        none_keys_string = ', '.join(none_keys)
        logger.info("填槽未完成,未完成key有：%s", none_keys_string)
        slot_info["intent_strategy"] = "deny"
    else:
        slot_info["slot_values"] = slot_values
        # 根据意图强度来确认回复策略
        if query_intensity >= intent_threshold_config["accept"]:
            slot_info["intent_strategy"] = "accept"
        elif query_intensity >= intent_threshold_config["deny"]:
            slot_info["intent_strategy"] = "clarify"
        else:
            slot_info["intent_strategy"] = "deny"

    # 上述填槽完毕，准备查询数据库
    return get_answer(slot_info)


def get_answer(slot_info):
    """
    根据语义槽获取答案回复
    """
    cql_template = slot_info.get("cql_template")
    cql_template_vision = slot_info.get("cql_template_vision")
    reply_template = slot_info.get("reply_template")
    ask_template = slot_info.get("ask_template")
    slot_values = slot_info.get("slot_values")
    strategy = slot_info.get("intent_strategy")

    if not slot_values:
        logger.info("slot填充未完成，采用默认值返回")
        slot_info["replay_answer"] = slot_info.get("deny_response")
        return slot_info

    if strategy == "accept":
        cql = []
        if isinstance(cql_template, list):
            for cqlt in cql_template:
                cql.append(cqlt.format(**slot_values))
        else:
            cql = cql_template.format(**slot_values)

        cql_vision = cql_template_vision.format(**slot_values)
        answer = neo4j_searcher(cql)
        # 查询可视化vision
        visison_data = neo4j_searcher_vision(cql_vision)
        slot_info["visison_data"] = visison_data
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


def neo4j_searcher_vision(cql_vision):
    data = []
    links = []
    kgdata = graph.run(cql_vision).data()
    if not kgdata:
        return [data, links]
    count = 0
    for value in kgdata:
        count += 1
        relNode = value['type']
        Relid = value['Relid']
        pNode = value['p']
        qNode = value['q']
        plabel_ = value['plabel']
        qlabel_ = value['qlabel']
        if count == 1:
            data.append({'id': str(qNode.identity), 'name': qNode['name'], 'des': qNode['name'],
                         'category': CATEGORY_INDEX[qlabel_]})
        else:
            data.append({'id': str(pNode.identity), 'name': pNode['name'], 'des': pNode['name'],
                         'category': CATEGORY_INDEX[plabel_]})
        links.append(
            {'source': str(qNode.identity), 'target': str(pNode.identity), 'value': relNode,
             'id': str(Relid)})

    return {
        'data': data,
        'links': links
    }


def entity_link(mention, etype):
    """
    对于识别到的实体mention,如果其不是知识库中的标准称谓
    则对其进行实体链指，将其指向一个唯一实体（待实现）
    """
    return mention
