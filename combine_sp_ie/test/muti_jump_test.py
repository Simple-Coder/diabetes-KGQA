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


# 假设 similarity_function 用于计算相似度
class SimilarityModel(torch.nn.Module):
    def __init__(self, input_dim, entity_dim):
        super(SimilarityModel, self).__init__()
        self.fc_input = torch.nn.Linear(input_dim, entity_dim)

    def forward(self, input_embedding, entity_embedding):
        input_embedding_mapped = self.fc_input(input_embedding)
        # 假设 entity_embedding 也经过了相应的线性变换
        # ...

        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(input_embedding_mapped, entity_embedding, dim=0)
        return cos_sim


# 计算相似度得分
def calculate_similarity_scores(similarity_model, input_embedding, entity_embeddings):
    # 计算输入特征与实体嵌入之间的相似度得分
    # 返回得分列表
    scores = []
    for entity_embedding in entity_embeddings:
        score = similarity_model(input_embedding, entity_embedding)
        scores.append(score)
    return scores


def generate_answer(entity, relation):
    if relation.startswith("有"):
        return f"{entity}有{relation[2:]}。"
    elif relation.startswith("剂量"):
        return f"{entity}{relation}。"
    elif relation.startswith("频率"):
        return f"{entity}{relation}。"
    else:
        return f"{entity}{relation}"


def main():
    # 构建问题和知识图谱
    question = "对于患有高血压的患者，有哪些药物可以治疗？药物的剂量和频率是多少？"
    knowledge_graph = {
        "高血压": {"有药物可以治疗": "药物A"},
        "药物A": {"有剂量和频率": "剂量X每天一次"},
        "剂量X每天一次": {"有剂量和频率": "频率每天一次"}
        # 可以继续添加更多关系
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

            # 编码问题
            question_tokens = tokenizer.tokenize(question)
            question_inputs = tokenizer.encode_plus(question_tokens, add_special_tokens=True, return_tensors='pt')
            question_embedding = bert_model(**question_inputs).last_hidden_state.mean(dim=1)  # 使用平均池化

            # 构建输入特征，将问题编码与邻居实体/关系的嵌入向量拼接
            inputs = [question_embedding]
            entity_embeddings = []
            for relation, neighbor in neighbors.items():
                if relation.startswith("有"):
                    entity_embedding = bert_model(**tokenizer.encode_plus(neighbor, add_special_tokens=True,
                                                                          return_tensors='pt')).last_hidden_state.mean(
                        dim=1)
                    inputs.append(entity_embedding)
                    entity_embeddings.append(entity_embedding)

            # 融合输入特征
            fused_embedding = torch.cat(inputs, dim=1)

            # 初始化相似度模型
            input_dim = fused_embedding.shape[1]  # 输入特征的维度
            entity_dim = entity_embeddings[0].shape[1]  # 实体嵌入的维度
            similarity_model = SimilarityModel(input_dim, entity_dim)

            # 运行预测，根据得分选择下一个实体
            if entity_embeddings:
                scores = calculate_similarity_scores(similarity_model, fused_embedding, entity_embeddings)
                predicted_next_entity_idx = scores.index(max(scores))
                predicted_next_entity = list(neighbors.values())[predicted_next_entity_idx]
            else:
                predicted_next_entity = None

            # 更新当前实体和路径
            if predicted_next_entity:
                current_entity = predicted_next_entity
                path.append(current_entity)
            else:
                break

        # 根据路径生成回答
        response = "；".join([generate_answer(entity, relation) for entity, relation in zip(path, path[1:])])
        print("答案:", response)
        if path:
            detailed_answer = generate_detailed_answer(path)
            print("详细回答:", detailed_answer)
    else:
        print("无法识别问题中的实体")


def generate_detailed_answer(path):
    answer = ""
    for i in range(len(path) - 1):
        if i == 0:
            answer += f"{path[i]}可以{path[i + 1]}治疗；"
        elif path[i].startswith("药物") and path[i + 1].startswith("剂量"):
            answer += f"{path[i]}{path[i + 1]}；"
        elif path[i].startswith("剂量") and path[i + 1].startswith("频率"):
            answer += f"{path[i]}{path[i + 1]}。"
    return answer


if __name__ == "__main__":
    main()
