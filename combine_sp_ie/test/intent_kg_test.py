"""
Created by xiedong
@Date: 2023/8/16 12:57
"""


class IntentEntity:
    def __init__(self, name):
        self.name = name
        self.related_intents = []

    def add_related_intent(self, intent, relationship):
        self.related_intents.append((intent, relationship))


# 创建意图实体
intents = {
    "Symptom_Disease": IntentEntity("临床表现"),
    "Reason_Disease": IntentEntity("病因"),
    "ADE_Drug": IntentEntity("不良反应"),
    "Amount_Drug": IntentEntity("用药剂量"),
    "Anatomy_Disease": IntentEntity("部位"),
    "Class_Disease": IntentEntity("分期类型"),
    "Drug_Disease": IntentEntity("药品名称"),
    "Method_Drug": IntentEntity("用药方法"),
    "Duration_Drug": IntentEntity("持续时间"),
    "Frequency_Drug": IntentEntity("用药频率"),
    "Operation_Disease": IntentEntity("手术"),
    "Pathogenesis_Disease": IntentEntity("发病机制"),
    "Test_Disease": IntentEntity("检查方法"),
    "Test_items_Disease": IntentEntity("检查指标"),
    "Treatment_Disease": IntentEntity("非药治疗"),
}

# 建立意图之间的关联关系
intents["Test_Disease"].add_related_intent(intents["Drug_Disease"], "检查方法->疾病")
intents["Symptom_Disease"].add_related_intent(intents["Drug_Disease"], "临床表现->疾病")
intents["Treatment_Disease"].add_related_intent(intents["Drug_Disease"], "非药治疗->疾病")
intents["Drug_Disease"].add_related_intent(intents["Drug_Disease"], "药品名称->疾病")
intents["Anatomy_Disease"].add_related_intent(intents["Drug_Disease"], "部位->疾病")
intents["Reason_Disease"].add_related_intent(intents["Drug_Disease"], "病因->疾病")
intents["Pathogenesis_Disease"].add_related_intent(intents["Drug_Disease"], "发病机制->疾病")
intents["Operation_Disease"].add_related_intent(intents["Drug_Disease"], "手术->疾病")
intents["Class_Disease"].add_related_intent(intents["Drug_Disease"], "分期类型->疾病")
intents["Test_items_Disease"].add_related_intent(intents["Drug_Disease"], "检查指标->疾病")

intents["Frequency_Drug"].add_related_intent(intents["Drug_Disease"], "用药频率->药品名称")
intents["Duration_Drug"].add_related_intent(intents["Drug_Disease"], "持续时间->药品名称")
intents["Amount_Drug"].add_related_intent(intents["Drug_Disease"], "用药剂量->药品名称")
intents["Method_Drug"].add_related_intent(intents["Drug_Disease"], "用药方法->药品名称")
intents["ADE_Drug"].add_related_intent(intents["Drug_Disease"], "不良反应->药品名称")


# 示例查询
def find_related_intents(intent_name):
    if intent_name in intents:
        related_intents = intents[intent_name].related_intents
        return [(intent.name, relationship) for intent, relationship in related_intents]
    return None


# 查询临床表现相关的意图
symptom_intent = "Symptom_Disease"
related_intents = find_related_intents(symptom_intent)
print(f"与{symptom_intent}相关的意图：", related_intents)

# 查询药品名称相关的意图
drug_intent = "Drug_Disease"
related_intents = find_related_intents(drug_intent)
print(f"与{drug_intent}相关的意图：", related_intents)
