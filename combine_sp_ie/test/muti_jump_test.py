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


def calculate_similarity_scores(bert_model, question, candidate_texts):
    question_tokens = tokenizer.tokenize(question)
    question_inputs = tokenizer.encode_plus(question_tokens, add_special_tokens=True, return_tensors='pt')
    question_embedding = bert_model(**question_inputs).last_hidden_state.mean(dim=1)  # 使用平均池化

    candidate_embeddings = []
    for text in candidate_texts:
        text_tokens = tokenizer.tokenize(text)
        text_inputs = tokenizer.encode_plus(text_tokens, add_special_tokens=True, return_tensors='pt')
        text_embedding = bert_model(**text_inputs).last_hidden_state.mean(dim=1)
        candidate_embeddings.append(text_embedding)

    similarity_scores = []
    for embedding in candidate_embeddings:
        # 计算余弦相似度得分
        cos_sim = torch.nn.functional.cosine_similarity(question_embedding, embedding, dim=1)
        similarity_scores.append(cos_sim.item())  # 将相似度得分添加到列表中

    return similarity_scores


def select_next_relation_and_entities(question, neighbors, current_relation):
    next_relation = None
    next_entities = []

    max_score = -1  # 初始化最大得分
    for relation, next_entity_list in neighbors.items():
        if relation != current_relation and next_entity_list:
            # 将当前关系与下一步关系拼接，以区分不同关系的候选实体
            candidate_texts = [current_relation] + [relation] + next_entity_list
            similarity_scores = calculate_similarity_scores(bert_model, question, candidate_texts)

            # 选择与当前关系最相似的关系和实体
            current_max_score = max(similarity_scores)
            if current_max_score > max_score:
                max_score = current_max_score
                next_relation = relation
                next_entities = next_entity_list

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
        "高血压": {"治疗": ["药物A", "药物B"], "病因": ["病因1", "病因2"]},
        "药物A": {"剂量和频率": ["剂量X每天一次"], "不良反应": ["头痛", "恶心"]},
        "药物B": {"剂量和频率": ["剂量Y每天一次"], "不良反应": ["头痛", "胃痛"]},
        "剂量X每天一次": {"剂量和频率": ["频率每天一次"]},
        "剂量Y每天一次": {"剂量和频率": ["频率每天一次"]},
        # 可以继续添加更多关系 @
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
            next_relation, next_entities = select_next_relation_and_entities(question, neighbors, path[-1])

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

        # 在 main 函数中调用 generate_detailed_answer
        if path:
            response = "；".join([generate_answer(entity, relation) for entity, relation in zip(path, path[1:])])
            print("答案:", response)
            response_templates = {
                "治疗": "'{}' 的治疗方式一般是这样的：{}",
                "剂量和频率": "'{}' 的剂量和频率一般是这样的：{}",
                "不良反应": "'{}' 的不良反应可能包括：{}",
                # 在这里添加更多关系和对应的回复模板
            }
            detailed_answer = generate_detailed_answer(path, response_templates)
            print("详细回答:", detailed_answer)
    else:
        print("无法识别问题中的实体")


def generate_detailed_answer(path, response_templates):
    detailed_answers = []
    for i in range(len(path) - 1):
        current_relation = path[i]
        next_relation = path[i + 1]
        response_template = response_templates.get(current_relation, "'{}'的信息为：{}")
        detailed_answer = response_template.format(current_relation, next_relation)
        detailed_answers.append(detailed_answer)
    return "；".join(detailed_answers)


if __name__ == "__main__":
    main()
