"""
Created by xiedong
@Date: 2023/6/5 15:11
"""
from ..context.context_manager import ContextManager
from ..nlu.intent_recognizer import IntentRecognizer


class DialogManager:
    def __init__(self):
        self.context_manager = ContextManager()
        self.intent_recognizer = IntentRecognizer()

    def process_user_input(self, message):
        # 解析意图
        intent = self.intent_recognizer.recognize_intent(message)

        # 处理上下文
        context_data = self.context_manager.get_context(intent)
        if context_data:
            # 上下文存在，执行相应的操作
            response = self.handle_context(intent, context_data)
        else:
            # 上下文不存在，执行默认操作
            response = self.handle_default_intent(intent)

        return response

    def handle_context(self, intent, context_data):
        # TODO: 根据上下文数据执行相应的操作，并构建响应
        response = {
            "intent": intent,
            "context_data": context_data,
            # ...
        }
        return response

    def handle_default_intent(self, intent):
        # TODO: 根据默认意图执行相应的操作，并构建响应
        response = {
            "intent": intent,
            # ...
        }
        return response
