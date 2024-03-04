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
        "cql_template_vision": "MATCH(p:`临床表现`)-[r:Symptom_Disease]->(q:`疾病`) where q.name='{disease}' RETURN type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病的病症表现一般是这样的：\n",
        "ask_template": "您问的是疾病 '{disease}' 的症状表现吗？",
        "intent_strategy": "",
        "deny_response": "人类的语言太难了！！"
    },
    "reason_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`病因`) -[r:Reason_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`病因`) -[r:Reason_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病的原因是：\n",
        "ask_template": "您问的是疾病 '{disease}' 的原因吗？",
        "intent_strategy": "",
        "deny_response": "您说的我有点不明白，您可以换个问法问我哦~"
    },
    "ade_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`不良反应`) -[r:ADE_Drug]->(q:`药物名称`) where q.name='{drug}' return p.name",
        "cql_template_vision": "MATCH (p:`不良反应`) -[r:ADE_Drug]->(q:`药物名称`) where q.name='{drug}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{drug}' 药物的不良反应、副作用是：\n",
        "ask_template": "您问的是药物 '{drug}' 的不良反应、副作用吗？",
        "intent_strategy": "",
        "deny_response": "呃，我好像没太听懂，您可以换个问法问我哦~"
    },
    "amount_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`用药剂量`) -[r:Amount_Drug]->(q:`药物名称`) where q.name='{drug}' return p.name",
        "cql_template_vision": "MATCH (p:`用药剂量`) -[r:Amount_Drug]->(q:`药物名称`) where q.name='{drug}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{drug}' 药物的用药剂量是：\n",
        "ask_template": "您问的是药物 '{drug}' 的用药剂量吗？",
        "intent_strategy": "",
        "deny_response": "对不起，我听起来似乎有些不太明白，您可以换个问法问我哦~"
    },
    "anatomy_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`部位`) -[r:Anatomy_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`部位`) -[r:Anatomy_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病发病的部位有：\n",
        "ask_template": "您问的是疾病 '{disease}' 发病的部位吗？",
        "intent_strategy": "",
        "deny_response": "我不太明白您的话，您可以换个问法问我哦~"
    },
    "class_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`分期类型`) -[r:Class_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`分期类型`) -[r:Class_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病分期类型是：\n",
        "ask_template": "您问的是疾病 '{disease}' 分期类型吗？",
        "intent_strategy": "",
        "deny_response": "我没太明白您刚才的话，有些摸不着头脑，您可以换个问法问我哦~"
    },
    "drug_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`药品名称`) -[r:Drug_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`药品名称`) -[r:Drug_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": " '{disease}' 的治疗药物有以下：\n",
        "ask_template": "您是想问 '{disease}' 的治疗药物有哪些吗？",
        "intent_strategy": "",
        "deny_response": "额~似乎有点不理解您说的是啥呢~~"
    },
    "method_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`用药方法`) -[r:Method_Drug]->(q:`药品名称`) where q.name='{drug}' return p.name",
        "cql_template_vision": "MATCH (p:`用药方法`) -[r:Method_Drug]->(q:`药品名称`) where q.name='{drug}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{drug}' 药物的用药方法是：\n",
        "ask_template": "您问的是药物 '{drug}' 用药方法吗？",
        "intent_strategy": "",
        "deny_response": "我听起来有点糊涂，您可以换个问法问我哦~"
    },
    "duration_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`持续时间`) -[r:Duration_Drug]->(q:`药品名称`) where q.name='{drug}' return p.name",
        "cql_template_vision": "MATCH (p:`持续时间`) -[r:Duration_Drug]->(q:`药品名称`) where q.name='{drug}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{drug}' 药物的持续时间是：\n",
        "ask_template": "您问的是药物 '{drug}' 的持续时间吗？",
        "intent_strategy": "",
        "deny_response": "很抱歉，我听不太懂，您可以换个问法问我哦~"
    },
    "frequency_drug": {
        "slot_list": ["drug"],
        "slot_values": None,
        "cql_template": "MATCH (p:`用药频率`) -[r:Frequency_Drug]->(q:`药品名称`) where q.name='{drug}' return p.name",
        "cql_template_vision": "MATCH (p:`用药频率`) -[r:Frequency_Drug]->(q:`药品名称`) where q.name='{drug}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{drug}' 药物的用药频率是：\n",
        "ask_template": "您问的是药物 '{drug}' 的用药频率吗？",
        "intent_strategy": "",
        "deny_response": "很抱歉，我无法领会您的意思，您可以换个问法问我哦~"
    },
    "operation_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`手术`) -[r:operation_disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`手术`) -[r:operation_disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病需要的手术是：\n",
        "ask_template": "您问的是疾病 '{disease}' 需要做什么手术吗？",
        "intent_strategy": "",
        "deny_response": "哦~我有点难以理解您所说的内容，您可以换个问法问我哦~"
    },
    "pathogenesis_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`发病机制`) -[r:Pathogenesis_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`发病机制`) -[r:Pathogenesis_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病的发病机制是：\n",
        "ask_template": "您问的是疾病 '{disease}' 的发病机制吗？",
        "intent_strategy": "",
        "deny_response": "很抱歉，我理解起来有点困难，您可以换个问法问我哦~"
    },
    "test_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`检查方法`) -[r:Test_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`检查方法`) -[r:Test_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病的检查方法有：\n",
        "ask_template": "您问的是疾病 '{disease}' 的检查方法吗？",
        "intent_strategy": "",
        "deny_response": "嗯，我有点迷糊，您可以换个问法问我哦~"
    },
    "test_items_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`检查指标`) -[r:Test_items_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`检查指标`) -[r:Test_items_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病的检查指标有：\n",
        "ask_template": "您问的是疾病 '{disease}' 的检查指标吗？",
        "intent_strategy": "",
        "deny_response": "对不起，我好像没太理解您说的话，请您再换个方式表达一下哦~"
    },
    "treatment_disease": {
        "slot_list": ["disease"],
        "slot_values": None,
        "cql_template": "MATCH (p:`非药治疗`) -[r:Treatment_Disease]->(q:`疾病`) where q.name='{disease}' return p.name",
        "cql_template_vision": "MATCH (p:`非药治疗`) -[r:Treatment_Disease]->(q:`疾病`) where q.name='{disease}' return type(r) AS type,id(r) as Relid,p,q,labels(p)[0] as plabel,labels(q)[0] as qlabel",
        "reply_template": "'{disease}' 疾病的非药治疗方式有：\n",
        "ask_template": "您问的是疾病 '{disease}' 的非药治疗方式吗？",
        "intent_strategy": "",
        "deny_response": "我不太理解您刚才所说的内容，您可以换个问法问我哦~"
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
        "您好呀",
        "我是智能医疗诊断机器人，有什么可以帮助您吗",
        "hi，您好，您可以叫我小谢",
        "您好，您可以问我一些关于疾病诊断的问题哦"
    ],
    "goodbye": [
        "再见，很高兴为您服务",
        "bye",
        "再见，感谢使用我的服务",
        "再见啦，祝您健康"
    ],
    "deny": [
        "很抱歉没帮到您",
        "I am sorry",
        "那您可以试着问我其他问题哟"
    ],
    "isbot": [
        "我是小谢，您的智能健康顾问",
        "您可以叫我小谢哦~",
        "我是医疗诊断机器人小谢"
    ],
}

# 意图强度配置
intent_threshold_config = {
    "accept": 0.9,
    "deny": 0.4
}

CATEGORY_INDEX = {'检查项目': 0, '科室': 1, '疾病': 2, '药品': 3,
                  '食物': 4, '药企': 5, '症状': 6,
                  '医生': 7, '医疗机构': 8}

DEFAULT_CHAT = {
    '请问擅长治疗2型糖尿病的医院有哪些?'
}
DEFAULT_CHAT_ABSWER = "擅长治疗'2型糖尿病'的医院有:"+"""
    河北北方学院附属第一医院、宁波大学附属第一医院、北京大学第一医院
    """

DEFAULT_MH_DATA = {
    "data": [
        {
            "category": 2,
            "des": "糖尿病",
            "id": "1",
            "name": "糖尿病"
        },
        {
            "category": 7,
            "des": "励丽",
            "id": "2",
            "name": "励丽"
        },
        {
            "category": 7,
            "des": "杨建梅",
            "id": "3",
            "name": "杨建梅"
        },
        {
            "category": 7,
            "des": "谷君",
            "id": "4",
            "name": "谷君"
        },
        {
            "category": 8,
            "des": "河北北方学院附属第一医院",
            "id": "5",
            "name": "河北北方学院附属第一医院"
        },
        {
            "category": 8,
            "des": "宁波大学附属第一医院",
            "id": "6",
            "name": "宁波大学附属第一医院"
        },
        {
            "category": 8,
            "des": "北京大学第一医院",
            "id": "7",
            "name": "北京大学第一医院"
        }
    ],
    "links": [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "value": "skilled_in"
        },
        {
            "id": "2",
            "source": "1",
            "target": "3",
            "value": "skilled_in"
        },
        {
            "id": "3",
            "source": "1",
            "target": "4",
            "value": "skilled_in"
        },
        {
            "id": "4",
            "source": "2",
            "target": "5",
            "value": "work_on"
        },
        {
            "id": "5",
            "source": "3",
            "target": "6",
            "value": "work_on"
        },
        {
            "id": "6",
            "source": "4",
            "target": "7",
            "value": "work_on"
        }
    ]
}

# CATEGORY_INDEX = {'疾病': 0, '疾病分期类型': 1, '病因': 2, '发病机制': 3,
#                   '临床表现': 4, '检查方法': 5, '检查指标': 6,
#                   '检查指标值': 7, '药物名称': 8, '用药频率': 9, '用药剂量': 10,
#                   '用药方法': 11, '非药治疗': 12, '手术': 13, '不良反应': 14,
#                   '部位': 15, '程度': 16, '持续时间': 17}

from enum import Enum


class AnswerEnum(Enum):
    # 2个均为闲聊：取第一个回答
    ANSWER_ALL_GOSSIP = 1
    # 2个均为诊断：合并答案
    ANSWER_ALL_MEDICAL = 2
    # 2个均为澄清：合并澄清
    ANSWER_ALL_CLARIFY = 3
    # 2个均为未知：返回学习中
    ANSWER_ALL_OTHERS = 4

    # 1个闲聊，一个诊断：合并诊断
    ANSWER_GOSSIP_MEDICAL = 5
    # 1个闲聊，一个澄清：合并澄清
    ANSWER_GOSSIP_CLARIFY = 6
    # 1个闲聊，一个未知：优先闲聊
    ANSWER_GOSSIP_OTHERS = 7

    # 1个诊断，一个闲聊：只处理诊断
    ANSWER_MEDICAL_GOSSIP = 8
    # 1个诊断，一个澄清：先诊断，再澄清
    ANSWER_MEDICAL_CLARIFY = 9
    # 1个诊断，一个未知：只诊断
    ANSWER_MEDICAL_OTHERS = 10

    # 一个澄清，一个闲聊：只处理澄清
    ANSWER_CLARIY_GOSSIP = 11
    # 一个澄清，一个诊断：合并诊断后澄清
    ANSWER_CLARIFY_MEDICAL = 12
    # 一个澄清，一个未知：只处理澄清
    ANSWER_CLARIFY_OTHERS = 13

    # 一个未知，一个闲聊：学习中
    ANSWER_OTHER_GOSSIP = 14
    # 一个未知，一个澄清：学习中
    ANSWER_OTHER_CLARIFY = 15
    # 一个未知，一个诊断：学习中
    ANSWER_OTHER_MEDICAL = 16
