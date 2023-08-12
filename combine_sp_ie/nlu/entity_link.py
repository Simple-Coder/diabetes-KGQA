"""
Created by xiedong
@Date: 2023/8/7 15:43
"""
from fuzzywuzzy import fuzz
from combine_sp_ie.kg.neo4j_client import Neo4jClient


class EntityLinkService():
    def __init__(self):
        self.neo4j_Client = Neo4jClient()

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

    def entity_links(self, ner_result):
        if not ner_result or len(ner_result) == 0:
            return None
        # TODO：待实现
        return ner_result
