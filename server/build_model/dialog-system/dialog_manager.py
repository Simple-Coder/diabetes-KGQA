"""
Created by xiedong
@Date: 2023/6/5 12:38
对话管理模块，负责协调其他模块的工作，并处理多轮对话的逻辑
"""
from nlu import IntentParser
from response_generator import ResponseGenerator
from context_manager import ContextManager


class DialogManager:
    def __init__(self):
        self.intent_parser = IntentParser()
        self.response_generator = ResponseGenerator()
        self.context_manager = ContextManager()

    def process_user_input(self, user_input):
        intent = self.intent_parser.parse_intent(user_input)
        system_response = self.response_generator.generate_response(intent, self.context_manager.context)
        self.context_manager.update_context(intent)
        return system_response
