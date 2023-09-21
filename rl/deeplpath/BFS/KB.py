# 定义一个知识库类（KB），用于存储实体之间的关系路径
class KB(object):
    def __init__(self):
        self.entities = {}

    # 添加关系到知识库中
    def addRelation(self, entity1, relation, entity2):
        if entity1 in self.entities:
            self.entities[entity1].append(Path(relation, entity2))
        else:
            self.entities[entity1] = [Path(relation, entity2)]

    # 获取从实体出发的所有关系路径
    def getPathsFrom(self, entity):
        return self.entities.get(entity, [])

    # 移除两个实体之间的关系路径
    def removePath(self, entity1, entity2):
        for idx, path in enumerate(self.entities.get(entity1, [])):
            if path.connected_entity == entity2:
                del self.entities[entity1][idx]
                break
        for idx, path in enumerate(self.entities.get(entity2, [])):
            if path.connected_entity == entity1:
                del self.entities[entity2][idx]
                break

    # 随机选择两个实体之间的中间实体
    def pickRandomIntermediatesBetween(self, entity1, entity2, num):
        from random import choice
        intermediates = set()
        entities = list(self.entities.keys())  # 将实体键转换为列表

        # 检查是否有足够的实体可供选择
        if num > len(entities) - 2:
            raise ValueError('选择的中间实体数量大于可能的数量', 'num_entities: {}'.format(len(entities)),
                             'num_intermediates: {}'.format(num))

        while len(intermediates) < num:
            intermediate = choice(entities)
            if intermediate not in intermediates and intermediate != entity1 and intermediate != entity2:
                intermediates.add(intermediate)

        return list(intermediates)

    def __str__(self):
        string = ""
        for entity in self.entities:
            string += entity + ','.join(str(x) for x in self.entities[entity])
            string += '\n'
        return string


# 定义路径类（Path），表示两个实体之间的关系
class Path(object):
    def __init__(self, relation, connected_entity):
        self.relation = relation
        self.connected_entity = connected_entity

    def __str__(self):
        return "\t{}\t{}".format(self.relation, self.connected_entity)

    __repr__ = __str__


if __name__ == "__main__":
    kb = KB()
    kb.addRelation("A", "knows", "B")
    kb.addRelation("B", "friends_with", "C")
    kb.addRelation("A", "likes", "D")
    kb.addRelation("D", "knows", "E")  # 添加更多关系以扩展知识库
    kb.addRelation("C", "dislikes", "F")
    kb.addRelation("F", "hates", "G")

    print("知识库内容：")
    print(kb)

    intermediates = kb.pickRandomIntermediatesBetween("A", "G", 1)  # 选择一个中间实体
    if intermediates:
        print("\n随机选择的中间实体：")
        print(intermediates)
    else:
        print("\n没有足够的中间实体可供选择。")
