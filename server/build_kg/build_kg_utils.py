"""
Created by xiedong
@Date: 2023/3/22 21:59
"""
import json

from tqdm import tqdm
from py2neo import Graph


# py2neo==2020.1.1

class DiabetesExtractor():
    def __init__(self):
        self.graph = Graph(
            host="127.0.0.1",
            http_port=7474,
            user="neo4j",
            password="123456")

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
        # 本人写增，未定义
        self.ade_disease = []  # 不良反应->疾病

    def extract_triples(self, data_path):
        print("从json文件中转换抽取三元组开始")
        with open(data_path, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                data_json = json.loads(line, strict=False)

                # 处理每个句子
                sentences = data_json['sentences']
                for sentence in sentences:
                    # 实体集合
                    entities = sentence['entities']
                    entiDict = {}
                    for entity in entities:

                        entity_type_ = entity['entity_type']
                        entity_id_ = entity['entity_id']
                        entity_name_ = entity['entity']

                        entiDict[entity_id_] = entity_name_

                        # 疾病实体
                        if "Disease" == entity_type_:
                            self.diseases.append(entity_name_)
                        # 药物名称
                        if "Drug" == entity_type_:
                            self.drugs.append(entity_name_)
                        # 不良反应
                        if "ADE" == entity_type_:
                            self.ades.append(entity_name_)
                        # 发病机制
                        if "Pathogenesis" == entity_type_:
                            self.pathogenesis.append(entity_name_)
                        # 用药剂量
                        if "Amount" == entity_type_:
                            self.amounts.append(entity_name_)
                        # 持续时间
                        if "Duration" == entity_type_:
                            self.duration.append(entity_name_)
                        # 用药方法
                        if "Method" == entity_type_:
                            self.methods.append(entity_name_)
                        # 手术
                        if "Operation" == entity_type_:
                            self.operations.append(entity_name_)
                        # 部位
                        if "Anatomy" == entity_type_:
                            self.anatomys.append(entity_name_)
                        # 病因
                        if "Reason" == entity_type_:
                            self.reasons.append(entity_name_)
                        # 非药治疗
                        if "Treatment" == entity_type_:
                            self.treatments.append(entity_name_)
                        # 检查方法
                        if "Test" == entity_type_:
                            self.tests.append(entity_name_)
                        # 用药频率
                        if "Frequency" == entity_type_:
                            self.frequency.append(entity_name_)
                        # 疾病分期类型
                        if "Class" == entity_type_:
                            self.classes.append(entity_name_)
                        # 程度
                        if "Level" == entity_type_:
                            self.levels.append(entity_name_)
                        # 检查指标值
                        if "Test_Value" == entity_type_:
                            self.test_values.append(entity_name_)
                        # 临床表现
                        if "Symptom" == entity_type_:
                            self.symptoms.append(entity_name_)
                        # 检查指标
                        if "Test_items" == entity_type_:
                            self.test_items.append(entity_name_)

                    # 关系集合
                    relations = sentence['relations']
                    for relation in relations:
                        relation_type_ = relation['relation_type']
                        head_entity_id = relation['head_entity_id']
                        tail_entity_id = relation['tail_entity_id']
                        head_entity_name = entiDict[head_entity_id]
                        tail_entity_name = entiDict[tail_entity_id]
                        # 调整关系
                        # head_entity_name = entiDict[tail_entity_id]
                        # tail_entity_name = entiDict[head_entity_id]
                        # 药品名称->疾病
                        if "Drug_Disease" == relation_type_:
                            self.drug_disease.append([head_entity_name, "Drug_Disease", tail_entity_name])
                        # 分期类型->疾病
                        if "Class_Disease" == relation_type_:
                            self.class_disease.append([head_entity_name, "Class_Disease", tail_entity_name])
                        # 用药剂量->药品名称
                        if "Amount_Drug" == relation_type_:
                            self.amount_drug.append([head_entity_name, "Amount_Drug", tail_entity_name])
                        # 持续时间->药品名称
                        if "Duration_Drug" == relation_type_:
                            self.duration_drug.append([head_entity_name, "Duration_Drug", tail_entity_name])
                        # 检查方法->疾病
                        if "Test_Disease" == relation_type_:
                            self.test_disease.append([head_entity_name, "Test_Disease", tail_entity_name])
                        # 用药方法->药品名称
                        if "Method_Drug" == relation_type_:
                            self.method_drug.append([head_entity_name, "Method_Drug", tail_entity_name])
                        # 部位->疾病
                        if "Anatomy_Disease" == relation_type_:
                            self.anatomy_disease.append([head_entity_name, "Anatomy_Disease", tail_entity_name])
                        # 手术->疾病
                        if "Operation_Disease" == relation_type_:
                            self.operation_disease.append([head_entity_name, "Operation_Disease", tail_entity_name])
                        # 不良反应->药品名称
                        if "ADE_Drug" == relation_type_:
                            self.ade_drug.append([head_entity_name, "ADE_Drug", tail_entity_name])
                        # 病因->疾病
                        if "Reason_Disease" == relation_type_:
                            self.reason_disease.append([head_entity_name, "Reason_Disease", tail_entity_name])
                            # 发病机制->疾病
                        if "Pathogenesis_Disease" == relation_type_:
                            self.pathogenesis_disease.append(
                                [head_entity_name, "Pathogenesis_Disease", tail_entity_name])
                        # 非药治疗->疾病
                        if "Treatment_Disease" == relation_type_:
                            self.treatment_disease.append([head_entity_name, "Treatment_Disease", tail_entity_name])
                        # 药品名称->疾病
                        if "Drug_Disease" == relation_type_:
                            self.drug_disease.append([head_entity_name, "Drug_Disease", tail_entity_name])
                        # 用药频率->药品名称
                        if "Frequency_Drug" == relation_type_:
                            self.frequency_drug.append([head_entity_name, "Frequency_Drug", tail_entity_name])
                        # 检查指标->疾病
                        if "Test_items_Disease" == relation_type_:
                            self.test_items_disease.append([head_entity_name, "Test_items_Disease", tail_entity_name])
                        # 临床表现->疾病
                        if "Symptom_Disease" == relation_type_:
                            self.symptom_disease.append([head_entity_name, "Symptom_Disease", tail_entity_name])
                            # self.symptom_disease.append([tail_entity_name, "Symptom_Disease", head_entity_name])
                        # 不良反应->疾病
                        if "ADE_Disease" == relation_type_:
                            self.ade_disease.append([head_entity_name, "ADE_Disease", tail_entity_name])

        print("从json文件中转换抽取三元组结束")
    def write_nodes(self, entities, entities_type):
        print("写入:" + entities_type + "：实体开始\n")
        for node in tqdm(set(entities)):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entities_type, entity_name=node.replace("'", "")
            )
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)
        print("写入:" + entities_type + "：实体结束\n")

    def write_edges(self, triples, head_type, tail_type):
        print("写入{0}关系".format(triples[0][1]))
        for head, relation, tail in tqdm(triples):
            cql = """MATCH(p:{head_type}),(q:{tail_type})
            WHERE p.name='{head}' AND q.name='{tail}'
            MERGE (p)-[r:{relation}]->(q)""".format(
                head_type=head_type, tail_type=tail_type, head=head.replace("'", ""),
                tail=tail.replace("'", ""), relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def create_entities(self):
        self.write_nodes(self.diseases, '疾病')
        self.write_nodes(self.classes, '疾病分期类型')
        self.write_nodes(self.reasons, '病因')
        self.write_nodes(self.pathogenesis, '发病机制')
        self.write_nodes(self.symptoms, '临床表现')
        self.write_nodes(self.tests, '检查方法')
        self.write_nodes(self.test_items, '检查指标')
        self.write_nodes(self.test_values, '检查指标值')
        self.write_nodes(self.drugs, '药物名称')
        self.write_nodes(self.frequency, '用药频率')
        self.write_nodes(self.amounts, '用药剂量')
        self.write_nodes(self.methods, '用药方法')
        self.write_nodes(self.treatments, '非药治疗')
        self.write_nodes(self.operations, '手术')
        self.write_nodes(self.ades, '不良反应')
        self.write_nodes(self.anatomys, '部位')
        self.write_nodes(self.levels, '程度')
        self.write_nodes(self.duration, '持续时间')

    def create_relations(self):
        self.write_edges(self.test_disease, '检查方法', '疾病')  # 检查方法->疾病
        self.write_edges(self.symptom_disease, '临床表现', '疾病')  # 临床表现->疾病
        self.write_edges(self.treatment_disease, '非药治疗', '疾病')  # 非药治疗->疾病
        self.write_edges(self.drug_disease, '药品名称', '疾病')  # 药品名称->疾病
        self.write_edges(self.anatomy_disease, '部位', '疾病')  # 部位->疾病
        self.write_edges(self.reason_disease, '病因', '疾病')  # 病因->疾病
        self.write_edges(self.pathogenesis_disease, '发病机制', '疾病')  # 发病机制->疾病
        self.write_edges(self.operation_disease, '手术', '疾病')  # 手术->疾病
        self.write_edges(self.class_disease, '分期类型', '疾病')  # 分期类型->疾病
        self.write_edges(self.test_items_disease, '检查指标', '疾病')  # 检查指标->疾病

        self.write_edges(self.frequency_drug, '用药频率', '药品名称')  # 用药频率->药品名称
        self.write_edges(self.duration_drug, '持续时间', '药品名称')  # 持续时间->药品名称
        self.write_edges(self.amount_drug, '用药剂量', '药品名称')  # 用药剂量->药品名称
        self.write_edges(self.method_drug, '用药方法', '药品名称')  # 用药方法->药品名称
        self.write_edges(self.ade_drug, '不良反应', '药品名称')  # 不良反应->药品名称


# self.write_edges(self.rels_department, '科室', '科室')


if __name__ == '__main__':
    # cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    # data_path = os.path.join(cur_dir, './data/kg_data/1.json')

    data_path = "../data/kg/clear_data/clear.json"

    extractor = DiabetesExtractor()
    extractor.extract_triples(data_path)

    # 创建实体节点
    extractor.create_entities()

    # 创建关系
    extractor.create_relations()

    print()
'''
发现实体记录：{"Drug":"Drug","ADE":"ADE","Disease":"Disease","Pathogenesis":"Pathogenesis","Amount":"Amount","Duration":"Duration","Method":"Method","Operation":"Operation","Anatomy":"Anatomy","Reason":"Reason","Treatment":"Treatment","Test":"Test","Frequency":"Frequency","Class":"Class","Level":"Level","Test_Value":"Test_Value","Symptom":"Symptom","Test_items":"Test_items"}
发现关系记录：{"Amount_Drug":"Amount_Drug","Duration_Drug":"Duration_Drug","Test_Disease":"Test_Disease","Method_Drug":"Method_Drug","Anatomy_Disease":"Anatomy_Disease","Operation_Disease":"Operation_Disease","ADE_Drug":"ADE_Drug","Reason_Disease":"Reason_Disease","Pathogenesis_Disease":"Pathogenesis_Disease","ADE_Disease":"ADE_Disease","Treatment_Disease":"Treatment_Disease","Drug_Disease":"Drug_Disease","Frequency_Drug":"Frequency_Drug","Test_items_Disease":"Test_items_Disease","Symptom_Disease":"Symptom_Disease","Class_Disease":"Class_Disease"}
'''
