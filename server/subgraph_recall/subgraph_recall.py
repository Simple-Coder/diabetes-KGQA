"""
Created by xiedong
@Date: 2023/8/2 21:51
"""
from py2neo import Graph
import torch
from transformers import BertTokenizer, BertModel
from relation import translate_relation

graph = Graph(
    host="127.0.0.1",
    http_port=7474,
    user="neo4j",
    password="123456")

bert_dir = 'hfl/chinese-bert-wwm-ext'


# bert_dir = 'D:\dev\PycharmProjects\diabetes-KGQA\server\chinese-bert-wwm-ext'


class RecallSubGraphAnswer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        self.model = BertModel.from_pretrained(bert_dir)

    def encode_query(self, query):
        inputs = self.tokenizer.encode_plus(query, add_special_tokens=True, return_tensors='pt')
        outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)  # 平均池化操作
        return query_embedding

    def encode_subgraph(self, subgraph):
        node_embedding = self.model(
            **self.tokenizer(subgraph["node"], add_special_tokens=True, return_tensors='pt')).last_hidden_state.mean(
            dim=1)
        relationship_embedding = self.model(**self.tokenizer(subgraph["relationship"], add_special_tokens=True,
                                                             return_tensors='pt')).last_hidden_state.mean(dim=1)
        related_node_embedding = self.model(**self.tokenizer(subgraph["related_node"], add_special_tokens=True,
                                                             return_tensors='pt')).last_hidden_state.mean(dim=1)

        return node_embedding, relationship_embedding, related_node_embedding

    def retrieve_subgraphs(self, query):
        entities = ["糖尿病"]

        subgraphs = []
        for entity in entities:
            cypher_query = "MATCH (n)-[r]-(m) WHERE n.name CONTAINS $entity OR m.name CONTAINS $entity RETURN n, type(r) AS relationship, m"
            result = graph.run(cypher_query, {"entity": entity})
            for record in result:
                subgraph = {
                    "node": record["n"]["name"],
                    "relationship": record["relationship"],
                    "related_node": record["m"]["name"]
                }
                subgraphs.append(subgraph)
        return subgraphs

    def rank_answers(self, query_embedding, subgraphs):
        ranked_subgraphs = sorted(subgraphs, key=lambda x: self.calculate_similarity(query_embedding, x["embedding"]),
                                  reverse=True)
        return ranked_subgraphs

    def calculate_similarity(self, query_embedding, subgraph_triplet):
        node_embedding, relationship_embedding, related_node_embedding = subgraph_triplet

        # 计算加权平均嵌入
        node_weight, rel_weight, related_node_weight = 0.4, 0.3, 0.3
        subgraph_embedding = node_weight * node_embedding + rel_weight * relationship_embedding + related_node_weight * related_node_embedding

        # 计算余弦相似度
        similarity = torch.cosine_similarity(query_embedding, subgraph_embedding, dim=1)
        return similarity

    def answer_question(self, query):
        query_embedding = self.encode_query(query)
        subgraphs = self.retrieve_subgraphs(query)

        # 遍历subgraphs，生成子图embedding和原始子图信息
        limit = 1
        subgraphs_with_embedding = []
        for subgraph in subgraphs:
            limit += 1
            if limit == 100:
                break
            subgraph_embedding = self.encode_subgraph(subgraph)
            subgraphs_with_embedding.append({"embedding": subgraph_embedding, "info": subgraph})

        ranked_subgraphs = self.rank_answers(query_embedding, subgraphs_with_embedding)

        # 提取排序后的子图的原始信息，即node，relationship和related_node
        sorted_subgraphs_info = []
        for ranked_subgraph in ranked_subgraphs:
            subgraph_info = ranked_subgraph["info"]
            sorted_subgraphs_info.append(subgraph_info)

        return sorted_subgraphs_info


def test():
    entities = ["糖尿病"]
    relations = ["Symptom_Disease", "Reason_Disease"]  # 这里可以设置为None或空列表来测试兼容性

    subgraphs = []
    for entity in entities:
        if not relations:
            cypher_query = f"MATCH (n)-[r]-(m) WHERE n.name CONTAINS $entity OR m.name CONTAINS $entity RETURN n, type(r) AS relationship, m"
            result = graph.run(cypher_query, {"entity": entity})
        else:
            relation_clause = "|".join(relations)
            cypher_query = f"MATCH (n)-[r:{relation_clause}]-(m) WHERE n.name CONTAINS $entity OR m.name CONTAINS $entity RETURN n, type(r) AS relationship, m"
            result = graph.run(cypher_query, {"entity": entity})

        for record in result:
            subgraph = {
                "node": record["n"]["name"],
                # "relationship": record["relationship"],
                "relationship": translate_relation(record["relationship"]),
                "related_node": record["m"]["name"]
            }
            subgraphs.append(subgraph)

    print(subgraphs)


def testAnswer():
    from collections import defaultdict

    # 模拟示例数据
    answers = [
        {'node': '2型糖尿病', 'relationship': 'Pathogenesis_Disease', 'related_node': '胰岛素分泌相对不足'},
        {'node': '2型糖尿病', 'relationship': 'Pathogenesis_Disease', 'related_node': '胰岛β细胞功能已明显衰竭'},
        {'node': '2型糖尿病', 'relationship': 'Pathogenesis_Disease', 'related_node': '后期β细胞功能衰竭'},
        {'node': '2型糖尿病', 'relationship': 'Pathogenesis_Disease', 'related_node': '胰岛素分泌不足'},
        {'node': '2型糖尿病', 'relationship': 'Pathogenesis_Disease', 'related_node': '胰岛素作用障碍'},
        {'node': '2型糖尿病', 'relationship': 'Pathogenesis_Disease', 'related_node': '胰岛素分泌绝对不足'},
    ]

    # 将相同关系的三元组合并
    merged_answers = defaultdict(list)
    for answer in answers:
        key = (answer['node'], answer['relationship'])
        merged_answers[key].append(answer['related_node'])

    # 根据合并后的数据生成回答字符串
    formatted_answers = []
    for key, values in merged_answers.items():
        node, relationship = key
        related_nodes = ', '.join(values)
        formatted_answers.append(f"{node}的{relationship}有{related_nodes}")

    # 打印合并后的回答
    for answer in formatted_answers:
        print(answer)


if __name__ == '__main__':
    testAnswer()
    print()
    test()
    print()

    entities = ['糖尿病']
    query = "请问糖尿病有哪些临床表现"
    sub_graph = RecallSubGraphAnswer()
    answers = sub_graph.answer_question(query)
    for answer in answers:
        print(f"{answer['node']} - {answer['relationship']} - {answer['related_node']}")

    from collections import defaultdict

    merged_answers = defaultdict(list)
    # 根据合并后的数据生成回答字符串
    formatted_answers = []
    for key, values in merged_answers.items():
        node, relationship = key
        related_nodes = ', '.join(values)
        formatted_answers.append(f"{node}的{relationship}有{related_nodes}")

    # 打印合并后的回答
    for answer in formatted_answers:
        print(answer)
