"""
Created by xiedong
@Date: 2023/8/4 15:14
"""
relation_map = {
    "Symptom_Disease": "临床表现",
    "Reason_Disease": "病因",
    "ADE_Drug": "不良反应",
    "Amount_Drug": "用药剂量",
    "Anatomy_Disease": "部位",
    "Class_Disease": "分期类型",
    "Drug_Disease": "药品名称",
    "Method_Drug": "用药方法",
    "Duration_Drug": "持续时间",
    "Frequency_Drug": "用药频率",
    "operation_disease": "手术",
    "Pathogenesis_Disease": "发病机制",
    "Test_Disease": "检查方法",
    "Test_items_Disease": "检查指标",
    "Treatment_Disease": "非药治疗",
}
# 创建反向映射字典
reverse_relation_map = {v: k for k, v in relation_map.items()}


class Relations():

    def get_en_relations(self):
        return list(relation_map.keys())

    def get_cn_relations(self):
        return list(reverse_relation_map.keys())

    def translate_relation(self, relation):
        # 实现关系类型的翻译逻辑，你可以自行定义关系类型到中文的映射
        return relation_map.get(relation, relation), reverse_relation_map.get(relation, relation)

    def find_related_intents(self, intent_name):
        if intent_name in intents:
            related_intents = intents[intent_name].related_intents
            return [(intent.name, relationship) for intent, relationship in related_intents]
        return None


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
intents["Symptom_Disease"].add_related_intent(intents["Drug_Disease"], "临床表现->药品名称")
intents["Reason_Disease"].add_related_intent(intents["Drug_Disease"], "病因->药品名称")
intents["ADE_Drug"].add_related_intent(intents["Drug_Disease"], "不良反应->药品名称")
intents["Amount_Drug"].add_related_intent(intents["Drug_Disease"], "用药剂量->药品名称")
intents["Anatomy_Disease"].add_related_intent(intents["Drug_Disease"], "部位->药品名称")
intents["Class_Disease"].add_related_intent(intents["Drug_Disease"], "分期类型->药品名称")
intents["Drug_Disease"].add_related_intent(intents["Drug_Disease"], "药品名称->疾病")
intents["Method_Drug"].add_related_intent(intents["Drug_Disease"], "用药方法->药品名称")
intents["Duration_Drug"].add_related_intent(intents["Drug_Disease"], "持续时间->药品名称")
intents["Frequency_Drug"].add_related_intent(intents["Drug_Disease"], "用药频率->药品名称")
intents["Operation_Disease"].add_related_intent(intents["Drug_Disease"], "手术->药品名称")
intents["Pathogenesis_Disease"].add_related_intent(intents["Drug_Disease"], "发病机制->药品名称")
intents["Test_Disease"].add_related_intent(intents["Drug_Disease"], "检查方法->药品名称")
intents["Test_items_Disease"].add_related_intent(intents["Drug_Disease"], "检查指标->药品名称")
intents["Treatment_Disease"].add_related_intent(intents["Drug_Disease"], "非药治疗->药品名称")


def translate_relation(relation):
    # 实现关系类型的翻译逻辑，你可以自行定义关系类型到中文的映射
    relation_map = {
        "Symptom_Disease": "临床表现",
        "Reason_Disease": "病因",
        "ADE_Drug": "不良反应",
        "Amount_Drug": "用药剂量",
        "Anatomy_Disease": "部位",
        "Class_Disease": "分期类型",
        "Drug_Disease": "药品名称",
        "Method_Drug": "用药方法",
        "Duration_Drug": "持续时间",
        "Frequency_Drug": "用药频率",
        "operation_disease": "手术",
        "Pathogenesis_Disease": "发病机制",
        "Test_Disease": "检查方法",
        "Test_items_Disease": "检查指标",
        "Treatment_Disease": "非药治疗",
    }

    # 创建反向映射字典
    reverse_relation_map = {v: k for k, v in relation_map.items()}

    return relation_map.get(relation, relation), reverse_relation_map.get(relation, relation)


relations_map_service = Relations()
# 测试翻译到中文
# translated_relation_cn, translated_relation_en = translate_relation("Symptom_Disease")

if __name__ == '__main__':
    relations = Relations()
    cn_relations = relations.get_cn_relations()
    print(cn_relations)

    en_relations = relations.get_en_relations()
    print(en_relations)

    translated_relation_cn, translated_relation_en = relations.translate_relation("临床表现")

    print("翻译到中文:", translated_relation_cn)
    print("翻译回英文:", translated_relation_en)

    # 查询临床表现相关的意图
    symptom_intent = "Symptom_Disease"
    related_intents = relations.find_related_intents(symptom_intent)
    print(f"与{symptom_intent}相关的意图：", related_intents)

    # 查询药品名称相关的意图
    drug_intent = "Drug_Disease"
    related_intents = relations.find_related_intents(drug_intent)
    print(f"与{drug_intent}相关的意图：", related_intents)
