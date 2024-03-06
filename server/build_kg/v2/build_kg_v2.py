"""
Created by xiedong
@Date: 2024/3/6 11:16
"""
import os

def txt_to_dict(directory):
    data_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.readlines()
                for line in content:
                    key, value = line.strip().split(',')
                    data_dict[key] = value
                    data_dict[value] = key
    return data_dict


def read_triples(filepath,entities_dict):
    result = []
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for line in content:
            triple= line.strip().split(',')
            head_key = triple[0]
            relation_key = triple[1]
            tail_key = triple[2]

            head = entities_dict[head_key]
            relation = entities_dict[relation_key]
            tail = entities_dict[tail_key]


            result.append([head, relation, tail])

    return result



if __name__ == '__main__':

    directory = './entities'
    # 读取实体、关系
    entities_dict = txt_to_dict(directory)
    # 读取三元组
    filepath = './triples.txt'
    triples = read_triples(filepath, entities_dict)
    print()


