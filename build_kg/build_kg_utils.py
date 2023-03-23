"""
Created by xiedong
@Date: 2023/3/22 21:59
"""


class DiabetesExtractor(object):
    def __init__(self):
        super.__init__()

        # 18类节点
        self.diseases = []  # 疾病
        self.classes = []  # 疾病分期类型
        self.reasons = []  # 病因
        self.pathogenesis = []  # 发病机制
        self.symptoms = []  # 临床表现
        self.tests = []  # 检查方法
        self.test_items = []  # 检查指标
        self.test_values = []  # 检查指标值
        self.drugs = []  # 药物名称
        self.frequency = []  # 用药频率
        self.amounts = []  # 用药剂量
        self.methods = []  # 用药方法
        self.treatments = []  # 非药治疗
        self.operations = []  # 手术
        self.ades = []  # 不良反应
        self.anatomys = []  # 部位
        self.levels = []  # 程度
        self.duration = []  # 持续时间

        # 15类实体关系
        self.test_disease = []  # 检查方法->疾病
        self.symptom_disease = []  # 临床表现->疾病
        self.treatment_disease = []  # 非药治疗->疾病
        self.drug_disease = []  # 药品名称->疾病
        self.anatomy_disease = []  # 部位->疾病
        self.reason_disease = []  # 病因->疾病
        self.pathogenesis_disease = []  # 发病机制->疾病
        self.operation_disease = []  # 手术->疾病
        self.class_disease = []  # 分期类型->疾病
        self.test_items_disease = []  # 检查指标->疾病

        self.frequency_drug = []  # 用药频率->药品名称
        self.duration_drug = []  # 持续时间->药品名称
        self.amount_drug = []  # 用药剂量->药品名称
        self.method_drug = []  # 用药方法->药品名称
        self.ade_drug = []  # 不良反应->药品名称


if __name__ == '__main__':
    pass
