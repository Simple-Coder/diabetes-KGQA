"""
Created by xiedong
@Date: 2023/6/5 15:31
"""
import json

from websocket_server import WebsocketServer
from muti_server.utils.user_question import QuestionInfo, AnswerInfo
from muti_server.utils.logger_conf import my_log
from muti_server.dm.dialog_context import DialogContext
from muti_server.nlu.nlu import NLU, SemanticInfo
from muti_server.nlg.nlg import NLG
import muti_server.dm.dialogue_state_tracking as dst
import muti_server.dm.dialogue_policy_optimization as dpo
from muti_server.utils.json_utils import json_str

log = my_log.logger


class RobotWebSocketHandler:
    def __init__(self, args):
        self.args = args
        self.dialogue_tracker = dst.DialogueStateTracker(args)
        self.dialog_policy_optimizer = dpo.DialoguePolicyOptimizer(args)
        self.nlu = NLU(args)
        self.nlg = NLG(args)

    def new_client(self, client, server):
        client_id = client['id']
        # user_context = DialogContext(room_id)
        # self.dialogue_tracker.add_context(room_id, user_context)
        log.info("客户端建立连接完成, 客户端id:{},上下文初始化成功...".format(client_id))
        log.info("客户端建立连接完成, 客户端id:{}".format(client_id))

    def client_left(self, client, server):
        client_id = client['id']
        # self.dialogue_tracker.remove_context(client_id)
        # log.info("客户端断开连接，客户端id：【{}】,上下文移除成功".format(client_id))
        log.info("客户端断开连接，客户端id：【{}】".format(client_id))

    def message_received(self, client, server, message):
        # 将用户输入封装为对象，后续使用
        question_info = self.convert_message(client, message)
        room_id = question_info.room_id
        user_name = question_info.user_name
        log.info("接收到客户端:[{}]，发来的的信息:{}".format(room_id, message))
        try:
            user_context = DialogContext(room_id)
            self.dialogue_tracker.add_context(room_id, user_context)
            log.info("roomId:{},上下文初始化成功...".format(room_id))


            # 1、NLU 模块处理用户输入
            semantic_info = self.nlu.predict(question_info.userQuestion)
            log.info("对应query：{},正在进行nlu识别意图与槽位阶段...识别意图结果:{}，识别槽位结果:{}".format(
                question_info.userQuestion, json_str(semantic_info.get_intent_infos()),
                json_str(semantic_info.get_entities())))

            # 2、DM 模块处理
            # 使用对话状态追踪模块更新对话状态
            log.info("对应query：{},正在进行dst更新用户上下文...".format(question_info.userQuestion))
            self.dialogue_tracker.update_context_sematic_info(room_id, semantic_info)

            # 3、NLG 模块处理
            log.info("对应query：{},正在进行nlg回答用户...".format(question_info.userQuestion))
            dialog_context = self.dialogue_tracker.get_context(room_id)
            self.nlg.generate_response(client, server, dialog_context, question_info)

        except Exception as e:
            log.error("服务器异常：{}".format(e))
            user_name = user_name
            room_id = question_info.room_id
            answer_info = AnswerInfo(room_id, user_name, '服务器异常啦~请找小谢查看！', answer_type=-1)
            server.send_message(client, json.dumps(answer_info.to_dict()))
            # server.send_message(client, '服务器异常啦~请找小谢查看！')

    @staticmethod
    def convert_message(client, message):
        query_json = json.loads(message)
        query_username = query_json['username']
        query_text = query_json['query']
        roomId = query_json['roomId']

        question_info = QuestionInfo(client['id'])
        question_info.user_name = query_username
        question_info.user_question = query_text
        question_info.room_id = roomId

        return question_info


class RobotWebsocketServer:
    def __init__(self, args):
        self.args = args
        self.server = None
        self.port = args.port

    def start(self):
        # 创建 WebSocket 服务器实例，并设置事件处理函数
        self.server = WebsocketServer(port=self.port)
        handler = RobotWebSocketHandler(self.args)
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
