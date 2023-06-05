"""
Created by xiedong
@Date: 2023/6/5 12:39
上下文管理模块，负责维护对话的上下文信息。
"""


class ContextManager:
    def __init__(self):
        self.context = {}

    def update_context(self, intent):
        # 根据意图更新上下文
        pass
