"""
Created by xiedong
@Date: 2023/6/6 21:16
"""


class DialogueStateTracker:
    def __init__(self):
        self.dialogue_state = {}

    def update_dialogue_state(self, intent, entities):
        pass

    # 在实际应用中，根据意图和实体更新对话状态
    # 更新 self.dialogue_state
    def get_dialogue_state(self):
        return self.dialogue_state
