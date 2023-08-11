import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class RelationMatchingModel(nn.Module):
    def __init__(self, bert_dir, num_layers=3):
        super(RelationMatchingModel, self).__init__()
        self.num_layers = num_layers
        self.bert = BertModel.from_pretrained(bert_dir)
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        self.relation_classifier = nn.Linear(self.bert.config.hidden_size * 3, 1)

    def forward(self, input_queries, main_entities, candidate_relations):
        # 编码输入的查询、主实体和候选关系
        input_queries_ids = self.tokenizer(input_queries, padding=True, truncation=True, return_tensors="pt")[
            "input_ids"]
        main_entities_ids = self.tokenizer(main_entities, padding=True, truncation=True, return_tensors="pt")[
            "input_ids"]
        candidate_relations_ids = \
        self.tokenizer(candidate_relations, padding=True, truncation=True, return_tensors="pt")["input_ids"]

        # 使用BERT编码输入的查询、主实体和候选关系
        encoded_queries = self.bert(input_queries_ids)[0]
        main_entity_embeddings = self.bert(main_entities_ids)[0][:, 0, :]
        relation_embeddings = self.bert(candidate_relations_ids)[0][:, 0, :]

        # 分层剪枝：仅保留顶部k层
        encoded_queries = encoded_queries[:, -self.num_layers:, :]

        # 扩展主实体和候选关系的维度以匹配encoded_queries
        main_entity_embeddings_expanded = main_entity_embeddings.unsqueeze(1).expand(-1, encoded_queries.shape[1], -1)
        relation_embeddings_expanded = relation_embeddings.unsqueeze(0).expand(main_entity_embeddings_expanded.shape[0],
                                                                               -1, -1)

        # 结合查询、主实体和关系的嵌入
        combined_representation = torch.cat(
            (encoded_queries, main_entity_embeddings_expanded, relation_embeddings_expanded), dim=-1)

        relation_scores = self.relation_classifier(combined_representation).squeeze(-1)
        sorted_relations = sorted(zip(candidate_relations, relation_scores.tolist()), key=lambda x: x[1], reverse=True)
        # return relation_scores
        return sorted_relations


if __name__ == '__main__':
    # 示例使用
    bert_dir = 'D:\dev\PycharmProjects\diabetes-KGQA\server\chinese-bert-wwm-ext'
    model = RelationMatchingModel(bert_dir, num_layers=3)

    input_queries = ["法国的首都是什么？", "谁写了哈利波特？"]
    main_entities = ["法国", "谁"]
    candidate_relations = ["的首都是", "写了", "由谁写"]
    sorted_relations = model(input_queries, main_entities, candidate_relations)
    # print(sorted_relations)
    for relation, score in sorted_relations:
        print(f"关系: {relation}，匹配分数: {score}")
