"""
Created by xiedong
@Date: 2023/6/5 21:13
"""
import time
import uuid


class QuestionInfo:
    __slots__ = (
        "clientId",
        "userName",
        "userQuestion",
        "questionUuid",
        "currentTimestamp"
    )

    def __init__(self, clientId):
        self.clientId = clientId
        self.questionUuid = uuid.uuid4()
        self.currentTimestamp = int(time.time())

    @property
    def user_name(self):
        return self.userName

    @user_name.setter
    def user_name(self, userName):
        self.userName = userName

    @property
    def user_question(self):
        return self.userQuestion

    @user_question.setter
    def user_question(self, userQuestion):
        self.userQuestion = userQuestion
