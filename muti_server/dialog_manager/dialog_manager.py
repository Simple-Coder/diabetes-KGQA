"""
Created by xiedong
@Date: 2023/6/5 15:32
"""

from muti_server.context.context_manager import ContextManager
from muti_server.nlu_module.nlu import NLU

from muti_server.utils.logger_conf import my_log

log = my_log.logger


class DialogManager:
    def __init__(self):
        self.context_manager = ContextManager()
        self.nlu = NLU()
        self.dialog_state = {}  # 对话状态

    def update_state(self, intents, entities):
        # 更新对话状态的逻辑
        self.dialog_state["intents"] = intents
        self.dialog_state["entities"] = entities
        return self.dialog_state

    def generate_response(self, dialog_state):
        # 根据对话状态生成系统回复的逻辑
        response = "这是一个系统回复"  # 假设系统回复为固定文本
        return response

    def process_user_input(self, question_info):
        user_id = question_info.clientId
        user_name = question_info.user_name
        message = question_info.user_question
        uuid = question_info.questionUuid
        timestamp = question_info.currentTimestamp
        log.info("开始识别用户:{}【userId:{}】，输入:{},uuid:{},timestamp:{}".format(user_name, user_id, message, uuid,
                                                                          timestamp))

        # 自然语言理解
        intents = self.nlu.extract_intents(message)
        entities = self.nlu.extract_entities(message)

        # 获取当前对话的上下文
        # 对话管理
        dialog_state = self.update_state(intents, entities)
        response = self.generate_response(dialog_state)

        context = self.context_manager.get_context(user_id)

        # 处理上下文
        # ...

        # 构建响应
        response = {
            "intent": intent,
            "message": message,
            # ...
        }

        return response
