"""
Created by xiedong
@Date: 2023/4/22 20:00
"""
from sys import stdin

import random
from chat_config import gossip_corpus,DEFAULT_CHAT,DEFAULT_MH_DATA,DEFAULT_CHAT_ABSWER
import requests
import json
from modules import get_answer, medical_robot
from utils import load_user_dialogue_context, dump_user_dialogue_context
import flask


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
    visison_data = []
    reply = "唔~服务异常啦~"
    if predict is not None:
        user_intent = predict["intent"]
        confidence = predict["confidence"]
        user_slots = predict["slots"]
        print("识别出意图---：", user_intent)
        print("识别出意图强度---：", confidence)
        print("识别出槽位---：", user_slots)
        if user_intent in ["greet", "goodbye", "deny", "isbot"]:
            reply = gossip_robot(user_intent)
        elif user_intent == "accept":
            reply = load_user_dialogue_context(username)
            reply = reply.get("choice_answer")
        else:
            reply = medical_robot(user_intent, user_slots, confidence, username)
            if 'visison_data' in reply:
                visison_data = reply.get("visison_data")
            if msg in DEFAULT_CHAT:
                visison_data = DEFAULT_MH_DATA
                reply["replay_answer"] = DEFAULT_CHAT_ABSWER
            if reply["slot_values"]:
                dump_user_dialogue_context(username, reply)
            reply = reply.get("replay_answer")
    return {
        'answer': reply,
        'visison_data': visison_data
    }


def gossip_robot(intent):
    return random.choice(
        gossip_corpus.get(intent)
    )


if __name__ == '__main__':
    app = flask.Flask(__name__)


    @app.route("/service/api/answer", methods=["GET", "POST"])
    def predict():
        data = {}
        param = flask.request.get_json()
        print(param)
        text = param["text"]
        reply = text_reply('张三', text)

        data_data = {}
        data_data["reply"] = reply

        data["data"] = data_data
        data["code"] = 20000
        data["message"] = 'success'

        return flask.jsonify(data)


    app.run('0.0.0.0', 60063)

    # username = '张三'
    # while True:
    #     line = stdin.readline().strip()  # strip()去掉最后的回车或者是空格
    #     if line == '':
    #         break
    #     # 处理
    #     reply = text_reply(username, line)
    #     print("应答：", reply)
