# 导入所需模块，兼容Python 2和3
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
import random


# 定义BFS函数，用于执行广度优先搜索
def BFS(kb, entity1, entity2):
    # 创建一个用于跟踪已找到路径的对象
    res = foundPaths(kb)
    # 标记起始实体为已找到，前一个节点和关系为空
    res.markFound(entity1, None, None)
    q = Queue()
    q.put(entity1)
    while not q.empty():
        curNode = q.get()
        for path in kb.getPathsFrom(curNode):
            nextEntity = path.connected_entity
            connectRelation = path.relation
            if not res.isFound(nextEntity):
                q.put(nextEntity)
                res.markFound(nextEntity, curNode, connectRelation)
            if nextEntity == entity2:
                # 重新构建实体列表和路径列表以获得完整路径
                entity_list, path_list = res.reconstructPath(entity1, entity2)
                return (True, entity_list, path_list)
    return (False, None, None)


# 定义测试函数
def test():
    # 创建一个虚拟的知识库对象（kb），以便测试BFS函数
    class KnowledgeBase:
        def __init__(self):
            self.entities = {
                "A": [("B", "knows"), ("C", "likes")],
                "B": [("D", "friends_with")],
                "C": [("E", "dislikes")],
                "D": [("F", "siblings_with")],
                "E": [("F", "hates")],
                "F": []
            }

        def getPathsFrom(self, entity):
            return [Path(neighbor, relation) for neighbor, relation in self.entities[entity]]

    class Path:
        def __init__(self, connected_entity, relation):
            self.connected_entity = connected_entity
            self.relation = relation

    kb = KnowledgeBase()

    # 测试BFS函数
    result, entity_list, path_list = BFS(kb, "A", "F")
    if result:
        print("找到路径：")
        for i in range(len(entity_list)):
            print(entity_list[i], end="")
            if i < len(path_list):
                print(" --" + path_list[i] + "--> ", end="")
        print()
    else:
        print("未找到路径。")


# 定义用于跟踪已找到路径的类
class foundPaths(object):
    def __init__(self, kb):
        self.entities = {}
        for entity, relations in kb.entities.items():
            self.entities[entity] = (False, "", "")

    def isFound(self, entity):
        return self.entities[entity][0]

    def markFound(self, entity, prevNode, relation):
        self.entities[entity] = (True, prevNode, relation)

    def reconstructPath(self, entity1, entity2):
        entity_list = []
        path_list = []
        curNode = entity2
        while curNode != entity1:
            entity_list.append(curNode)
            path_list.append(self.entities[curNode][2])
            curNode = self.entities[curNode][1]
        entity_list.append(curNode)
        entity_list.reverse()
        path_list.reverse()
        return (entity_list, path_list)

    def __str__(self):
        res = ""
        for entity, status in self.entities.items():
            res += entity + "[{},{},{}]".format(status[0], status[1], status[2])
        return res


# 测试示例
if __name__ == "__main__":
    test()
