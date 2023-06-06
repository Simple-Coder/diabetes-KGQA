"""
Created by xiedong
@Date: 2023/6/6 16:00
参数异常
"""


class NoIntentsException(Exception):
    def __init__(self, message):
        self.message = message
