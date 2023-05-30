"""
Created by xiedong
@Date: 2023/5/30 12:44
"""


class NoIntentsException(Exception):
    def __init__(self, message):
        self.message = message
