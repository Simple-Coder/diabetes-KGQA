"""
Created by xiedong
@Date: 2023/8/16 12:57
"""


class Entity:
    def __init__(self, name):
        self.name = name
        self.dependencies = []

    def add_dependency(self, entity):
        self.dependencies.append(entity)

# 创建实体
entities = {
    "Disease": Entity("疾病"),
    "Drug": Entity("药物名称"),
    "ADE": Entity("不良反应"),
}

# 建立关联关系
entities["Drug"].add_dependency(entities["Disease"])
entities["ADE"].add_dependency(entities["Drug"])

# 示例关联关系
entities["Disease"].add_dependency(entities["Drug"])  # 将疾病与药物关联
entities["Drug"].add_dependency(entities["ADE"])  # 将药物与不良反应关联

# 示例查询
def find_related_entities(entity_name):
    if entity_name in entities:
        related_entities = entities[entity_name].dependencies
        return [entity.name for entity in related_entities]
    return None

# 查询疾病对应的药物
disease = "某疾病"
related_drugs = find_related_entities("Disease")
print(f"与疾病 {disease} 相关的药物：", related_drugs)

# 查询药物对应的不良反应
drug = "某药物"
related_ades = find_related_entities("Drug")
print(f"与药物 {drug} 相关的不良反应：", related_ades)
