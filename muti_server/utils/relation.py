"""
Created by xiedong
@Date: 2023/8/4 15:14
"""


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
        "Drug_Product": "生产厂商",
        "Food_Replace": "替代品",

        "symptom_disease": "临床表现",
        "disease_food_replace": "替代品",
    }

    # 创建反向映射字典
    reverse_relation_map = {v: k for k, v in relation_map.items()}

    return relation_map.get(relation, relation), reverse_relation_map.get(relation, relation)


# 测试翻译到中文
translated_relation_cn, translated_relation_en = translate_relation("Symptom_Disease")
print("翻译到中文:", translated_relation_cn)
print("翻译回英文:", translated_relation_en)

# 测试翻译到英文
translated_relation_cn, translated_relation_en = translate_relation("替代品")
print("翻译到中文:", translated_relation_cn)
print("翻译回英文:", translated_relation_en)
