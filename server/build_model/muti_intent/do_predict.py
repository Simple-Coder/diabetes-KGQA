"""
Created by xiedong
@Date: 2023/5/27 20:26
"""
import flask
from muti_config import Args
from muti_model import MutiJointModel
import torch
from muti_predict import Predictor
from transformers import logging

logging.set_verbosity_error()


class MutiPredictWrapper:
    def __init__(self, args):
        self.args = args
        # 加载模型
        self.model = MutiJointModel(self.args.seq_num_labels, self.args.token_num_labels)
        # 是否加载本地模型
        self.model.load_state_dict(torch.load(self.args.load_dir))

        #
        self.predictor = Predictor(self.model)

    def predict(self, text):
        return self.predictor.predict(text)


if __name__ == '__main__':
    args = Args()
    predict_wrapper = MutiPredictWrapper(args)

    app = flask.Flask(__name__)


    @app.route("/service/api/muti/predict", methods=["GET", "POST"])
    def predict():
        data = {}
        param = flask.request.get_json()
        print(param)
        text = param["text"]

        result = predict_wrapper.predict(text)

        data_data = {}

        # 1、按意图强度降序排序
        data_data["intents"] = sorted(result[0], key=lambda x: x[1], reverse=True)
        data_data["slots"] = result[1]

        data["data"] = data_data
        data["code"] = 20000
        data["message"] = 'success'

        return flask.jsonify(data)


    app.run('0.0.0.0', 60062)

    wrapper_predict = predict_wrapper.predict("请问二型糖尿病的临床表现是什么,需要吃什么药啊")
    print(wrapper_predict)
