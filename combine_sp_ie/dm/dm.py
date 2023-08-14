"""
Created by xiedong
@Date: 2023/8/7 12:42
"""
from combine_sp_ie.config.logger_conf import my_log
from muti_server.nlg.nlg_utils import fill_slot_info
from muti_server.utils.json_utils import json_str
from combine_sp_ie.kg.kgqa_processor import KGQAProcessor

log = my_log.logger


class DialogueStateTracker:
    def __init__(self):
        self.kgqa_processor = KGQAProcessor()
        self.contexts = {}

    def add_context(self, context_name, context_data):
        self.contexts[context_name] = context_data

    def get_context(self, context_name):
        return self.contexts.get(context_name)

    def remove_context(self, context_name):
        if context_name in self.contexts:
            del self.contexts[context_name]

    def update_context_sematic_info(self, client_id, semantic_info):
        try:
            """
            记录语义信息到上下文
            填槽
            查询neo4j记录每个意图的答案
            :param client_id:
            :param semantic_info:
            :return:
            """
            dialog_context = self.contexts.get(client_id)
            if not dialog_context:
                return

            intent_infos = semantic_info.get_intent_infos()
            entities = semantic_info.get_entities()

            if len(intent_infos) == 0:
                log.error("[dm] 意图识别为空，不处理查询neo4j")
                return

            for intent_info in intent_infos:
                intent = intent_info.get_intent()
                # strategy = intent_info.get_intent_strategy()
                strategy = intent_info.get_intent_enum()
                slot_info = fill_slot_info(intent, entities, dialog_context)
                slot_info = self.kgqa_processor.search(slot_info, strategy)

                if slot_info:
                    intent_info.set_answer_info(slot_info)

            # 填充语义信息
            dialog_context.set_current_semantic(semantic_info)
            log.info("[dst] update current sematic finish,infos:{}".format(json_str(semantic_info)))
        except Exception as e:
            log.error("[dst] update dialog context error:{}".format(e))


class DialoguePolicyOptimizer:
    def __init__(self):
        self.policy = {}

    def decide_action(self, dialogue_state):
        pass
        # 在实际应用中，根据对话状态决定系统的动作或回应
        # 根据 self.policy 和 dialogue_state 决定动作或回应

    def update_policy(self, reward):
        pass
        # 在实际应用中，根据奖励信号更新对话策略
        # 更新 self.policy

    def perform_action_and_get_reward(self, action):
        # 在实际应用中，根据动作执行对应的操作或回应
        # 并根据对话系统的目标和评估指标计算奖励信号
        # 返回奖励信号

        reward = self.calculate_reward(action)

        return reward

    def calculate_reward(self, action):
        # 根据对话系统的目标和评估指标计算奖励信号
        # 返回奖励信号
        # 可以根据具体需求定义自己的奖励计算函数

        # 计算奖励的逻辑
        return 'reward'
