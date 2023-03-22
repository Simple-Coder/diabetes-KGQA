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


if __name__ == '__main__':
    pass
