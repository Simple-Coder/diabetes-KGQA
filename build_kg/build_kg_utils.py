"""
Created by xiedong
@Date: 2023/3/22 21:59
"""
import json

from tqdm import tqdm
import os


class DiabetesExtractor():
    def __init__(self):
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

    def extract_triples(self, data_path):
        print("从json文件中转换抽取三元组")
        with open(data_path, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                data_json = json.loads(line, strict=False)

                # 处理每个句子
                sentences = data_json['sentences']
                for sentence in sentences:
                    # 实体集合
                    entities = sentence['entities']
                    # for entity in entities:


                    # 关系集合
                    relations = sentence['relations']

                    print()


if __name__ == '__main__':
    # cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    # data_path = os.path.join(cur_dir, './data/kg_data/1.json')

    data_path = "../data/clear_data/clear.json"

    extractor = DiabetesExtractor()
    extractor.extract_triples(data_path)

'''
发现实体记录：{"Drug":"Drug","ADE":"ADE","Disease":"Disease","Pathogenesis":"Pathogenesis","Amount":"Amount","Duration":"Duration","Method":"Method","Operation":"Operation","Anatomy":"Anatomy","Reason":"Reason","Treatment":"Treatment","Test":"Test","Frequency":"Frequency","Class":"Class","Level":"Level","Test_Value":"Test_Value","Symptom":"Symptom","Test_items":"Test_items"}
发现关系记录：{"Amount_Drug":"Amount_Drug","Duration_Drug":"Duration_Drug","Test_Disease":"Test_Disease","Method_Drug":"Method_Drug","Anatomy_Disease":"Anatomy_Disease","Operation_Disease":"Operation_Disease","ADE_Drug":"ADE_Drug","Reason_Disease":"Reason_Disease","Pathogenesis_Disease":"Pathogenesis_Disease","ADE_Disease":"ADE_Disease","Treatment_Disease":"Treatment_Disease","Drug_Disease":"Drug_Disease","Frequency_Drug":"Frequency_Drug","Test_items_Disease":"Test_items_Disease","Symptom_Disease":"Symptom_Disease","Class_Disease":"Class_Disease"}
'''
