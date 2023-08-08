"""
Created by xiedong
@Date: 2023/8/7 12:37

输入原始Query，输出Query理解结果

Query理解是KBQA的第一个核心模块，负责对句子的各个成分进行细粒度语义理解，其中两个最重要的模块是
1)实体识别和实体链接，输出问句中有意义的业务相关实体和类型，如商家名称、项目、设施、人群、时间等。
2)依存分析：以分词和词性识别结果为输入，识别问句的主实体、被提问信息、约束等。

实体识别是句法分析的重要步骤，我们先基于序列标注模型识别实体，再链接到数据库中的节点

"""

from combine_sp_ie.models.model_wrapper import ModelService
from combine_sp_ie.nlu.query_understand import QueryUnderstand
from combine_sp_ie.nlu.relation_recognize import RelationRecognize


class NLU():
    def __init__(self, args):
        self.args = args
        self.model_service = ModelService()
        # Query理解
        self.query_understand = QueryUnderstand()
        # 关系识别
        self.relation_recognize = RelationRecognize()

    def process_nlu(self, query):
        # 1、Query理解
        main_entity, intent, intent_conf, question_type, dependency_analysis \
            = self.query_understand.query_understanding(query)
        # 2、关系识别
        recognize_relation = self.relation_recognize.relation_recognition(query, intent, dependency_analysis)

        return main_entity, recognize_relation
