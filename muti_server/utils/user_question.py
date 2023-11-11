"""
Created by xiedong
@Date: 2023/6/5 21:13
"""
import json
import time
import uuid


class QuestionInfo:
    __slots__ = (
        "clientId",
        "userName",
        "userQuestion",
        "questionUuid",
        "roomId",
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
    def room_id(self):
        return self.roomId

    @room_id.setter
    def room_id(self, room_id):
        self.roomId = room_id

    @property
    def user_question(self):
        return self.userQuestion

    @user_question.setter
    def user_question(self, userQuestion):
        self.userQuestion = userQuestion


class AnswerInfo:
    def __init__(self, room_id, user_name, answer_text, answer_type=1):
        self.roomId = room_id
        self.userName = user_name
        self.answer = answer_text
        self.answer_type = answer_type

    def to_dict(self):
        return {
            'room_id': self.roomId,
            'user_name': self.userName,
            'answer': self.answer,
            'answer_type': self.answer_type
        }


if __name__ == '__main__':
    info = AnswerInfo(2, 3, 4, 5)

    # 使用 to_dict 方法将对象转换为字典
    json_string = json.dumps(info.to_dict())

    print(json_string)
