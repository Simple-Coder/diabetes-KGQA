"""
Created by xiedong
@Date: 2023/8/11 12:51
pip install fuzzywuzzy

"""
from combine_sp_ie.app import parse_arguments
from combine_sp_ie.kg.neo4j_client import Neo4jClient
from fuzzywuzzy import fuzz


class EntityLinker:
    def __init__(self, args):
        self.neo4j_Client = Neo4jClient(args)

    def link_entity(self, named_entity):
        # 使用Cypher查询在Neo4j中查找实体，并计算相似度
        query = f"MATCH (e) WHERE e.name CONTAINS '{named_entity}' RETURN e.name AS matched_entity"

        result = self.neo4j_Client.execute(query)

        max_similarity = 0
        best_match = None

        for record in result:
            entity_name = record["matched_entity"]
            similarity = fuzz.ratio(named_entity.lower(), entity_name.lower())

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = entity_name

        return best_match


# 示例用法
if __name__ == "__main__":
    args = parse_arguments()

    linker = EntityLinker(args)

    input_text = "糖尿病"
    named_entities = input_text.split()  # 简化示例，以空格分隔的单词作为命名实体

    for named_entity in named_entities:
        linked_entity = linker.link_entity(named_entity)
        if linked_entity:
            print(f"Linked '{named_entity}' to entity: '{linked_entity}'")
        else:
            print(f"No entity found for '{named_entity}'")
