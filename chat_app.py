"""
Created by xiedong
@Date: 2023/4/22 20:00
"""
from sys import stdin

import random
from chat_config import gossip_corpus
import requests
import json
from modules import get_answer, medical_robot
from utils import load_user_dialogue_context, dump_user_dialogue_context


def parse_text(text):
    url = 'http://127.0.0.1:60062/service/api/predict'
    data = {"text": text}
    headers = {'Content-Type': 'application/json;charset=utf8'}
    reponse = requests.post(url, data=json.dumps(data), headers=headers)
    if reponse.status_code == 200:
        reponse = json.loads(reponse.text)
        return reponse['data']
    else:
        return None


def text_reply(username, msg):
    predict = parse_text(msg)
    if predict is not None:
        user_intent = predict["intent"]
        user_slots = predict["slots"]
        print("识别出意图---：", user_intent)
        print("识别出槽位---：", user_slots)
        if user_intent in ["greet", "goodbye", "deny", "isbot"]:
            reply = gossip_robot(user_intent)
        elif user_intent == "accept":
            reply = load_user_dialogue_context(msg.User['NickName'])
            reply = reply.get("choice_answer")
        else:
            reply = medical_robot(user_intent, user_slots, username)
            if reply["slot_values"]:
                dump_user_dialogue_context(username, reply)
            reply = reply.get("replay_answer")
    else:
        reply = '服务异常啦~~~'
    return reply


def gossip_robot(intent):
    return random.choice(
        gossip_corpus.get(intent)
    )


if __name__ == '__main__':
    username = '张三'
    while True:
        line = stdin.readline().strip()  # strip()去掉最后的回车或者是空格
        if line == '':
            break
        # 处理
        reply = text_reply(username, line)
        print("应答：", reply)
