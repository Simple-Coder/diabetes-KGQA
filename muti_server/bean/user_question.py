"""
Created by xiedong
@Date: 2023/6/5 21:13
"""
import time
import uuid


class QuestionInfo:
    __slots__ = (
        "userId",
        "userQuestion",
        "questionUuid",
        "currentTimestamp"
    )

    def __init__(self):
        self.questionUuid = uuid.uuid4()
        self.currentTimestamp = int(time.time())

    @property
    def user_id(self):
        return self.userId

    @user_id.setter
    def user_id(self, userId):
        self.userId = userId

    @property
    def user_question(self):
        return self.userQuestion

    @user_question.setter
    def user_question(self, userQuestion):
        self.userQuestion = userQuestion
