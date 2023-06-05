"""
Created by xiedong
@Date: 2023/6/5 15:32
"""

from muti_server.context.context_manager import ContextManager
from muti_server.nlu.intent_recognizer import IntentRecognizer


class DialogManager:
    def __init__(self):
        self.context_manager = ContextManager()
        self.intent_recognizer = IntentRecognizer()

    def process_user_input(self, message):
        # 解析意图
        intent = self.intent_recognizer.recognize_intent(message)

        # 处理上下文
        # ...

        # 构建响应
        response = {
            "intent": intent,
            "message": message,
            # ...
        }

        return response
