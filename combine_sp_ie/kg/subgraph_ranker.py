"""
Created by xiedong
@Date: 2023/8/8 12:49
"""
import combine_sp_ie.config.wrapper as wrapper


class SubgraphRanker:
    def __init__(self, args):
        self.args = args

    def rank_and_filter_subgraphs(self, query, constraint, subgraphs):
        # 假设相关性分数已经计算好，可以根据实际情况替换为实际的分数计算方法
        sorted_subgraphs = sorted(subgraphs, key=lambda x: self.calculate_subgraph_score(query, x), reverse=True)

        filtered_subgraphs = [subgraph for subgraph in sorted_subgraphs if self.meets_constraint(subgraph, constraint)]

        # 选取Top1作为答案
        if filtered_subgraphs:
            return filtered_subgraphs[0]
        else:
            return None

    def calculate_subgraph_score(self, query, subgraph):
        # 根据查询和子图计算相关性分数，这里是示例，您可以根据实际情况进行计算
        score = some_scoring_function(query, subgraph)
        return score

    def meets_constraint(self, subgraph, constraint):
        # 根据约束条件判断子图是否满足约束，这里是示例，您可以根据实际情况进行判断
        return some_constraint_check(subgraph, constraint)
