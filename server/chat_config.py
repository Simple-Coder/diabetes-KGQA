"""
Created by xiedong
@Date: 2023/4/5 22:25
"""
semantic_slot = {
    "symptom_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        # "cql_template": "MATCH(p:疾病)-[r:has_symptom]->(q:症状) WHERE p.name='{Disease}' RETURN q.name",
        "cql_template": "MATCH(p:`临床表现`)-[r:Symptom_Disease]->(q:`疾病`) where q.name='{disease}' RETURN p.name",
        "reply_template": "'{disease}' 疾病的病症表现一般是这样的：\n",
        "ask_template": "您问的是疾病 '{disease}' 的症状表现吗？",
        "intent_strategy": "",
        "deny_response": "人类的语言太难了！！"
    },
    "reason_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`病因`) -[r:Reason_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "reply_template": "'{disease}' 疾病的原因是：\n",
        "ask_template": "您问的是疾病 '{disease}' 的原因吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "ade_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`不良反应`) -[r:ADE_Drug]->(q:`药物名称`) where q.name='{drug}' return p.name",
        "reply_template": "'{drug}' 药物的不良反应、副作用是：\n",
        "ask_template": "您问的是药物 '{drug}' 的不良反应、副作用吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "amount_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`用药剂量`) -[r:Amount_Drug]->(q:`药物名称`) where q.name='{drug}' return p.name",
        "reply_template": "'{drug}' 药物的用药剂量是：\n",
        "ask_template": "您问的是药物 '{drug}' 的用药剂量吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "anatomy_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`部位`) -[r:Anatomy_Disease]->(q:`疾病`) where q.name='{drug}' return p.name",
        "reply_template": "'{disease}' 疾病发病的部位是：\n",
        "ask_template": "您问的是疾病 '{disease}' 发病的部位吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "class_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`分期类型`) -[r:Class_Disease]->(q:`疾病`) where q.name='{drug}' return p.name",
        "reply_template": "'{disease}' 疾病分期类型是：\n",
        "ask_template": "您问的是疾病 '{disease}' 分期类型吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "drug_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`药品名称`) -[r:Drug_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "reply_template": " '{disease}' 的治疗药物有以下：\n",
        "ask_template": "您是想问 '{disease}' 的治疗药物有哪些吗？",
        "intent_strategy": "",
        "deny_response": "额~似乎有点不理解你说的是啥呢~~"
    },
    "method_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`用药方法`) -[r:Method_Drug]->(q:`药品名称`) where q.name='{drug}' return p.name",
        "reply_template": "'{drug}' 药物的用药方法是：\n",
        "ask_template": "您问的是药物 '{drug}' 用药方法吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "duration_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`持续时间`) -[r:Duration_Drug]->(q:`药品名称`) where q.name='{drug}' return p.name",
        "reply_template": "'{drug}' 药物的持续时间是：\n",
        "ask_template": "您问的是药物 '{drug}' 的持续时间吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "frequency_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`用药频率`) -[r:Frequency_Drug]->(q:`药品名称`) where q.name='{drug}' return p.name",
        "reply_template": "'{drug}' 药物的用药频率是：\n",
        "ask_template": "您问的是药物 '{drug}' 的用药频率吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "operation_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`手术`) -[r:operation_disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "reply_template": "'{disease}' 疾病需要的手术是：\n",
        "ask_template": "您问的是疾病 '{disease}' 需要做什么手术吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "pathogenesis_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`发病机制`) -[r:Pathogenesis_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "reply_template": "'{disease}' 疾病的发病机制是：\n",
        "ask_template": "您问的是疾病 '{disease}' 的发病机制吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "test_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`检查方法`) -[r:Test_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "reply_template": "'{disease}' 疾病的检查方法是：\n",
        "ask_template": "您问的是疾病 '{disease}' 的检查方法吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "test_items_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`检查指标`) -[r:Test_items_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "reply_template": "'{disease}' 疾病的检查指标有：\n",
        "ask_template": "您问的是疾病 '{disease}' 的检查指标吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "others": {
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
