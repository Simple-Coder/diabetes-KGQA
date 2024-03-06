"""
Created by xiedong
@Date: 2024/3/6 11:16
"""
import os
from tqdm import tqdm
from py2neo import Graph


def txt_to_dict(directory):
    data_dict1 = {}
    data_dict2 = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.readlines()
                for line in content:
                    key, value = line.strip().split(',')
                    data_dict1[key] = value
                    data_dict2[value] = key
    return data_dict1, data_dict2


def read_triples(graph, filepath, entities_dict):
    result = []
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for line in content:
            triple = line.strip().split(',')
            head_key = triple[0]
            relation_key = triple[1]
            tail_key = triple[2]

            type_head = fuzzy_match_dict(head_key)
            type_tail = fuzzy_match_dict(tail_key)
            if "" == type_head or "" == type_tail:
                continue
            head = entities_dict[head_key]
            relation = entities_dict[relation_key]
            tail = entities_dict[tail_key]

            write_edges(graph, head, relation, tail, type_head, type_tail)

            result.append([head, relation, tail])
    return result


def write_nodes(graph, node, entities_type):
    print("写入:" + node + "：实体开始\n")
    cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
        label=entities_type, entity_name=node.replace("'", "")
    )
    try:
        graph.run(cql)
    except Exception as e:
        print(e)
        print(cql)
    print("写入:" + node + "：实体结束\n")


def write_edges(graph, head, relation, tail, head_type, tail_type):
    print("写入{0}关系".format(relation))
    cql = """MATCH(p:{head_type}),(q:{tail_type})
    WHERE p.name='{head}' AND q.name='{tail}'
    MERGE (p)-[r:{relation}]->(q)""".format(
        head_type=head_type, tail_type=tail_type, head=head.replace("'", ""),
        tail=tail.replace("'", ""), relation=relation)
    try:
        graph.run(cql)
    except Exception as e:
        print(e)
        print(cql)


def fuzzy_match_dict(query):
    match_dict = {
        "check": "检查项目",
        "Depart": "科室",
        "disease": "疾病",
        "doctor": "医生",
        "Food": "食物",
        "Org": "医疗机构",
        "Medicine": "药物",
        "producer": "药企",
        "sympthom": "症状",
    }
    for key in match_dict:
        if query.startswith(key):
            return match_dict[key]
    return ""


if __name__ == '__main__':

    graph = Graph(host="127.0.0.1", http_port=7474, user="neo4j", password="123456")

    directory = './entities'
    # 读取实体、关系
    key_value, val_key = txt_to_dict(directory)
    # 实体写入
    for key, value in val_key.items():
        type = fuzzy_match_dict(key)
        if "" == type:
            continue
        write_nodes(graph, value, type)

    # 读取三元组
    filepath = './triples.txt'
    triples = read_triples(graph, filepath, val_key)

    # 关系写入

    print()
