"""
Created by xiedong
@Date: 2023/6/5 15:32
"""

from muti_server.context.context_manager import ContextManager
from muti_server.nlu.intent_recognizer import IntentRecognizer

from muti_server.utils.logger_conf import my_log

log = my_log.logger


class DialogManager:
    def __init__(self):
        self.context_manager = ContextManager()
        self.intent_recognizer = IntentRecognizer()

    def process_user_input(self, question_info):
        user_id = question_info.clientId
        user_name = question_info.user_name
        message = question_info.user_question
        uuid = question_info.questionUuid
        timestamp = question_info.currentTimestamp
        log.info("开始识别用户:{}【userId:{}】，输入:{},uuid:{},timestamp:{}".format(user_name, user_id, message, uuid,
                                                                                  timestamp))

        # 解析意图
        intent = self.intent_recognizer.recognize_intent(message)

        # 获取当前对话的上下文
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
