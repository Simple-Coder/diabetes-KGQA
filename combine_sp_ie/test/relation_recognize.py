"""
Created by xiedong
@Date: 2023/8/7 16:55
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class RelationMatchingModel(nn.Module):
    def __init__(self, num_layers=3):
        super(RelationMatchingModel, self).__init__()
        self.num_layers = num_layers
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.relation_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_queries, input_features, candidate_relations):
        # 使用BERT编码输入的查询
        encoded_queries = self.bert(input_queries)[0]

        # 分层剪枝：仅保留顶部k层
        encoded_queries = encoded_queries[:, -self.num_layers:, :]

        # 提取领域和句法信息
        domain_feature = input_features["domain"]
        syntax_feature = input_features["syntax"]

        # 结合领域和句法特征
        combined_features = torch.cat((domain_feature, syntax_feature), dim=-1)

        # 计算每个候选关系的匹配得分
        relation_scores = []
        for relation in candidate_relations:
            relation_embedding = self.bert(relation)[0][:, 0, :]  # 使用[CLS]标记的嵌入
            combined_representation = torch.cat((encoded_queries, relation_embedding, combined_features), dim=-1)
            relation_score = self.relation_classifier(combined_representation).squeeze(-1)
            relation_scores.append(relation_score)

        return relation_scores


if __name__ == '__main__':
    # 示例使用
    model = RelationMatchingModel(num_layers=3)

    input_queries = ["法国的首都是什么？", "谁写了哈利波特？"]
    input_features = {
        "domain": torch.tensor([[0.1, 0.2]]),  # 示例领域特征
        "syntax": torch.tensor([[0.3, 0.4]])  # 示例句法特征
    }

    candidate_relations = ["的首都是", "写了", "由谁写"]
    relation_scores = model(input_queries, input_features, candidate_relations)
    print(relation_scores)
