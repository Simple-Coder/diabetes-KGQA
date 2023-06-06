"""
Created by xiedong
@Date: 2023/6/6 21:52
"""


class NLG():
    def generate_response(self, client, server):

        server.send_message(client, '测试回答！')
