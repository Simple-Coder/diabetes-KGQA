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
    }
    return relation_map.get(relation, relation)