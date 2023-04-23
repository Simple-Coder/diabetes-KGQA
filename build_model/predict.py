"""
Created by xiedong
@Date: 2023/4/22 20:05
"""
import torch

from config import Args
from model import BertForIntentClassificationAndSlotFilling
from trainer import Trainer
from transformers import logging
from flask import request, jsonify
import flask

logging.set_verbosity_error()


class OKModel:
    def __init__(self, args):
        self.args = args
        # 加载模型
        self.model = BertForIntentClassificationAndSlotFilling(self.args)
        # 是否加载本地模型
        self.model.load_state_dict(torch.load(self.args.load_dir))
        # 训练器实例
        self.trainer = Trainer(self.model, self.args)

    def predict(self, text):
        return self.trainer.predict(text)


if __name__ == '__main__':
    app = flask.Flask(__name__)
    args = Args()
    ok_model = OKModel(args)


    @app.route("/service/api/predict", methods=["GET", "POST"])
    def predict():
        data = {"sucess": 0}
        result = None

        param = flask.request.get_json()
        print(param)
        text = param["text"]
        result = ok_model.predict(text)

        data_data = {}
        data_data["intent"] = result[0]
        data_data["confidence"] = result[1]
        data_data["slots"] = result[2]

        data["data"] = data_data
        data["sucess"] = 200

        return flask.jsonify(data)


    app.run('0.0.0.0', 60062)
    # # # 全局配置参数
    # args = Args()
    # # # 加载模型
    # # model = BertForIntentClassificationAndSlotFilling(args)
    # # # 是否加载本地模型
    # # model.load_state_dict(torch.load(args.load_dir))
    # # # 训练器实例
    # # trainer = Trainer(model, args)
    #
    # ok_model = OKModel(args)
    #
    # predict = ok_model.predict('请问二型糖尿病的临床表现是什么')
    #
    # print(predict)
    #
    # # trainer.predict('hi')
    # # trainer.predict('你是谁')
