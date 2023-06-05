"""
Created by xiedong
@Date: 2023/6/5 21:06
"""


class UserContext:
    __slots__ = [
        'user_id',  # 用户的唯一标识符，用于区分不同的用户
        'session_id',  # 用户当前会话的标识符，用于跟踪用户在会话中的交互。
        'username',  # 用户名，用户的可读性标识符。
        'last_interaction_time',  # 上一次用户与系统交互的时间戳，用于跟踪用户活动时间。
        'current_intent',  # 当前用户意图，表示用户当前的意图或请求。
        'previous_intent',  # 上一个用户意图，表示用户之前的意图或请求。
        'context_data'  # 存储其他与用户会话相关的上下文数据，如之前的对话历史、系统状态等。
    ]

    def __init__(self, user_id):
        self.user_id = user_id
        self.session_id = None
        self.username = None
        self.last_interaction_time = None
        self.current_intent = None
        self.previous_intent = None
        self.context_data = {}

    # def __init__(self, user_id, session_id, username):
    #     self.user_id = user_id
    #     self.session_id = session_id
    #     self.username = username
    #     self.last_interaction_time = None
    #     self.current_intent = None
    #     self.previous_intent = None
    #     self.context_data = {}

    def set_last_interaction_time(self, timestamp):
        self.last_interaction_time = timestamp

    def set_current_intent(self, intent):
        self.current_intent = intent

    def set_previous_intent(self, intent):
        self.previous_intent = intent

    def add_context_data(self, key, value):
        self.context_data[key] = value

    def get_context_data(self, key):
        return self.context_data.get(key)

    def remove_context_data(self, key):
        if key in self.context_data:
            del self.context_data[key]
