"""
Created by xiedong
@Date: 2023/6/5 15:31
"""
import json

import jsonpickle
from websocket_server import WebsocketServer
from muti_server.utils.context_manager import ContextManager
from muti_server.utils.user_question import QuestionInfo
from muti_server.utils.logger_conf import my_log
from muti_server.dm.dialog_context import DialogContext
from muti_server.nlu.nlu import NLU, SemanticInfo
from muti_server.nlg.nlg import NLG
import muti_server.dm.dialogue_state_tracking as dst
import muti_server.dm.dialogue_policy_optimization as dpo

log = my_log.logger


class RobotWebSocketHandler:
    def __init__(self):
        self.dialogue_tracker = dst.DialogueStateTracker()
        self.dialog_policy_optimizer = dpo.DialoguePolicyOptimizer()
        self.context_manager = ContextManager()
        self.nlu = NLU()
        self.nlg = NLG()

    def new_client(self, client, server):
        client_id = client['id']
        user_context = DialogContext(client_id)
        self.context_manager.add_context(client_id, user_context)
        log.info("客户端建立连接完成, 客户端id:{},上下文初始化成功...".format(client_id))

    def client_left(self, client, server):
        client_id = client['id']
        self.context_manager.remove_context(client_id)
        log.info("客户端断开连接，客户端id：【{}】,上下文移除成功".format(client_id))

    def message_received(self, client, server, message):
        try:
            log.info("接收到客户端:[{}]，发来的的信息:{}".format(client['id'], message))
            # 将用户输入封装为对象，后续使用
            question_info = self.convert_message(client, message)
            dialog_context = self.dialogue_tracker.get_dialogue_state()

            # 1、NLU 模块处理用户输入
            semantic_info = self.nlu.predict(message)
            log.info("对应query：{},正在进行nlu识别意图与槽位阶段...识别意图结果:{}，识别槽位结果:{}".format(
                question_info.userQuestion, jsonpickle.encode(semantic_info.get_intent_infos()),
                jsonpickle.encode(semantic_info.get_entities())))

            # DM 模块处理
            # 使用对话状态追踪模块更新对话状态
            # self.dialogue_tracker.update_dialogue_state(intents, entities)
            log.info("对应query：{},正在进行dst更新用户上下文...".format(question_info.userQuestion))

            log.info("对应query：{},正在进行nlg回答用户...".format(question_info.userQuestion))
            # NLG 模块处理
            self.nlg.generate_response(client, server, dialog_context)

        except Exception as e:
            log.error("服务器异常：{}".format(e))
            server.send_message(client, '服务器异常啦~请找小谢查看！')

    @staticmethod
    def convert_message(client, message):
        query_json = json.loads(message)
        query_username = query_json['username']
        query_text = query_json['query']

        question_info = QuestionInfo(client['id'])
        question_info.user_name = query_username
        question_info.user_question = query_text

        return question_info


class RobotWebsocketServer:
    def __init__(self, port):
        self.server = None
        self.port = port

    def start(self):
        # 创建 WebSocket 服务器实例，并设置事件处理函数
        self.server = WebsocketServer(port=self.port)
        handler = RobotWebSocketHandler()
        self.server.set_fn_new_client(handler.new_client)
        self.server.set_fn_client_left(handler.client_left)
        self.server.set_fn_message_received(handler.message_received)

        # 启动 WebSocket 服务器
        self.server.run_forever()


if __name__ == '__main__':
    # 创建 WebSocket 服务器
    server = RobotWebsocketServer(9999)
    print('启动test成功')
    server.start()
