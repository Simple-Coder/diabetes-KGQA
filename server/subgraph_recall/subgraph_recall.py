"""
Created by xiedong
@Date: 2023/8/2 21:51
"""
from py2neo import Graph
import torch
from transformers import BertTokenizer, BertModel

# match(n:`检查指标`) -[r:Test_items_Disease] ->(q:`疾病`) where q.name contains '病' return n,r,q limit 25
graph = Graph(
    host="127.0.0.1",
    http_port=7474,
    user="neo4j",
    password="123456")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# 加载NER模型（假设已训练好并保存在 ner_model 变量中）
# ner_model = load_ner_model()


def encode_query(query):
    inputs = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def encode_subgraph(subgraph):
    node_embedding = model(
        **tokenizer(subgraph["node"], add_special_tokens=True, return_tensors='pt')).last_hidden_state.mean(dim=1)
    relationship_embedding = model(
        **tokenizer(subgraph["relationship"], add_special_tokens=True, return_tensors='pt')).last_hidden_state.mean(
        dim=1)
    related_node_embedding = model(
        **tokenizer(subgraph["related_node"], add_special_tokens=True, return_tensors='pt')).last_hidden_state.mean(
        dim=1)
    subgraph_embedding = torch.cat([node_embedding, relationship_embedding, related_node_embedding], dim=1)
    return subgraph_embedding


# 1、查询子图
def retrieve_subgraphs(query):
    # 使用NER模型识别查询中的实体
    # entities = ner_model(query)
    entities = ["糖尿病"]

    # 根据实体查询知识图谱中的子图
    subgraphs = []
    for entity in entities:
        # 构建查询语句，根据实体的属性查询与之相关的子图
        cypher_query = "MATCH (n)-[r]-(m) WHERE n.name CONTAINS $entity OR m.name CONTAINS $entity RETURN n, r, m"
        result = graph.run(cypher_query, {"entity": entity["name"]})
        for record in result:
            subgraph = {
                "node": record["n"]["name"],
                "relationship": record["r"].type,
                "related_node": record["m"]["name"]
            }
            subgraphs.append(subgraph)
    return subgraphs


# 2、答案排序和输出
def rank_answers(query_embedding, subgraphs):
    ranked_subgraphs = sorted(subgraphs, key=lambda x: calculate_similarity(query_embedding, x), reverse=True)
    return ranked_subgraphs


def calculate_similarity(query_embedding, subgraph_embedding):
    return torch.cosine_similarity(query_embedding, subgraph_embedding)


def answer_question(query):
    query_embedding = encode_query(query)
    subgraphs = retrieve_subgraphs(query)
    ranked_subgraphs = rank_answers(query_embedding, subgraphs)
    return ranked_subgraphs


if __name__ == '__main__':
    query = "Where does Mary work?"
    answers = answer_question(query)
    for answer in answers:
        print(f"{answer['node']} - {answer['relationship']} - {answer['related_node']}")
