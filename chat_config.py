"""
Created by xiedong
@Date: 2023/4/5 22:25
"""
semantic_slot = {
    "symptom_disease": {
        "slot_list": ["Disease"],
        "slot_values": None,
        # "cql_template": "MATCH(p:疾病)-[r:has_symptom]->(q:症状) WHERE p.name='{Disease}' RETURN q.name",
        "cql_template": "MATCH(p:`临床表现`)-[r:Symptom_Disease]->(q:`疾病`) where q.name='{Disease}' RETURN p.name",
        "reply_template": "'{Disease}' 疾病的病症表现一般是这样的：\n",
        "ask_template": "您问的是疾病 '{Disease}' 的症状表现吗？",
        "intent_strategy": "",
        "deny_response": "人类的语言太难了！！"
    },
    "unrecognized": {
        "slot_values": None,
        "replay_answer": "非常抱歉，我还不知道如何回答您，我正在努力学习中~",
    }
}
# 闲聊数据配置
gossip_corpus = {
    "greet": [
        "hi",
        "你好呀",
        "我是智能医疗诊断机器人，有什么可以帮助你吗",
        "hi，你好，你可以叫我小谢",
        "你好，你可以问我一些关于疾病诊断的问题哦"
    ],
    "goodbye": [
        "再见，很高兴为您服务",
        "bye",
        "再见，感谢使用我的服务",
        "再见啦，祝你健康"
    ],
    "deny": [
        "很抱歉没帮到您",
        "I am sorry",
        "那您可以试着问我其他问题哟"
    ],
    "isbot": [
        "我是小谢，你的智能健康顾问",
        "你可以叫我小谢哦~",
        "我是医疗诊断机器人小谢"
    ],
}

# 意图强度配置
intent_threshold_config = {
    "accept": 0.9,
    "deny": 0.4
}
