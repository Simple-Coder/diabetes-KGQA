"""
Created by xiedong
@Date: 2023/6/6 17:31
"""


# 自然语言生成（NLG）模块
class NLG:
    def answer(self, client, server, user_context):
        # 生成自然语言文本的逻辑
        generated_text = 'response'  # 假设直接将系统回复作为生成的文本

        server.send_message(client, str(generated_text))
        return generated_text
