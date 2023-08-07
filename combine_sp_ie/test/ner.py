"""
Created by xiedong
@Date: 2023/8/7 15:40
"""
from ltp import LTP

def query_understanding(query):
    ltp = LTP()
    result = ltp.pipeline([query], tasks=["ner", "dep"])
    ner_results = result[0][0]
    dep_results = result[0][1]

    main_entity = None
    related_entities = []
    question_info = None
    constraints = []

    for idx, (entity_type, deprel) in enumerate(zip(ner_results, dep_results["label"])):
        entity = query[idx]
        if entity_type != "O":
            if main_entity is None:
                main_entity = entity
            else:
                related_entities.append(entity)
        if deprel == "HED":
            question_info = entity
        elif deprel == "COO":
            constraints.append(entity)

    return main_entity, related_entities, question_info, constraints

def entity_linking(entity):
    # 在这里实现将实体链接到数据库中的节点，得到节点id
    # 返回节点id
    return entity.lower() + "_id"  # 示例，将实体名转换为小写并加上后缀

# 测试示例
query = "故宫周末有学生票吗"
main_entity, related_entities, question_info, constraints = query_understanding(query)

main_entity_id = entity_linking(main_entity)
related_entities_ids = [entity_linking(entity) for entity in related_entities]

print("主实体:", main_entity, "节点ID:", main_entity_id)
print("关联实体:", related_entities, "节点IDs:", related_entities_ids)
print("问题信息:", question_info)
print("约束条件:", constraints)
