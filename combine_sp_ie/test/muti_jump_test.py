"""
Created by xiedong
@Date: 2023/8/13 20:42
"""
import torch
from transformers import BertTokenizer, BertModel
from combine_sp_ie.config.base_config import GlobalConfig

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(GlobalConfig.bert_dir)
bert_model = BertModel.from_pretrained(GlobalConfig.bert_dir)


class SimilarityModel(torch.nn.Module):
    def __init__(self, input_dim, entity_dim):
        super(SimilarityModel, self).__init__()
        self.fc_input = torch.nn.Linear(input_dim, entity_dim)

    def forward(self, input_embedding, entity_embedding):
        input_embedding_mapped = self.fc_input(input_embedding)
        cos_sim = torch.nn.functional.cosine_similarity(input_embedding_mapped, entity_embedding, dim=0)
        return cos_sim

def calculate_similarity_scores(bert_model, input_text, candidate_text_list):
    input_tokens = tokenizer.tokenize(input_text)
    input_inputs = tokenizer.encode_plus(input_tokens, add_special_tokens=True, return_tensors='pt')
    input_embedding = bert_model(**input_inputs).last_hidden_state.mean(dim=1)  # 使用平均池化

    candidate_embeddings = []
    for candidate_text in candidate_text_list:
        candidate_tokens = tokenizer.tokenize(candidate_text)
        candidate_inputs = tokenizer.encode_plus(candidate_tokens, add_special_tokens=True, return_tensors='pt')
        candidate_embedding = bert_model(**candidate_inputs).last_hidden_state.mean(dim=1)  # 使用平均池化
        candidate_embeddings.append(candidate_embedding)

    similarity_scores = []
    for candidate_embedding in candidate_embeddings:
        score = torch.nn.functional.cosine_similarity(input_embedding, candidate_embedding, dim=1)
        similarity_scores.append(score.item())

    return similarity_scores

def select_next_relation_and_entities(neighbors, current_relation):
    next_relation = None
    next_entities = []

    for relation, next_entity_list in neighbors.items():
        if relation != current_relation and next_entity_list:
            candidate_texts = [relation] + next_entity_list
            similarity_scores = calculate_similarity_scores(bert_model, current_relation, candidate_texts)
            max_score_idx = similarity_scores.index(max(similarity_scores))

            next_relation = relation
            next_entities = next_entity_list
            break

    return next_relation, next_entities

def generate_answer(entity, relation):
    if relation.endswith("剂量和频率"):
        return f"{entity}每天的{relation}是多少。"
    elif relation.endswith("剂量"):
        return f"{entity}的{relation}是多少。"
    elif relation.endswith("频率"):
        return f"{entity}的{relation}是多少。"
    else:
        return f"{entity}可以{relation}。"


def main():
    # 构建问题和知识图谱
    question = "对于患有高血压的患者，有哪些药物可以治疗？药物的剂量和频率是多少？"
    knowledge_graph = {
        "高血压": {"治疗": ["药物A"]},
        "药物A": {"剂量和频率": ["剂量X每天一次"]},
        "剂量X每天一次": {"剂量和频率": ["频率每天一次"]}
    }

    # 解析问题中的实体
    entities = ['高血压']  # 假设直接使用所有识别出的实体

    if entities:
        current_entity = entities[0]  # 使用第一个识别出的实体作为 current_entity
    else:
        current_entity = None

    if current_entity:
        # 多跳问答
        max_hops = 3
        path = [current_entity]

        for hop in range(max_hops):
            # 获取当前实体的邻居关系
            neighbors = knowledge_graph.get(current_entity, {})

            # 如果没有邻居，终止跳跃
            if not neighbors:
                break

            # 选择下一步关系和实体
            next_relation, next_entities = select_next_relation_and_entities(neighbors, path[-1])

            if next_relation and next_entities:
                # 更新当前实体和路径
                current_entity = next_entities[0]  # 假设只选择第一个候选实体
                path.append(next_relation)
                path.append(current_entity)
            else:
                break

        # 生成回答
        response = "；".join([generate_answer(entity, relation) for entity, relation in zip(path, path[1:])])
        print("答案:", response)
    else:
        print("无法识别问题中的实体")


if __name__ == "__main__":
    main()
