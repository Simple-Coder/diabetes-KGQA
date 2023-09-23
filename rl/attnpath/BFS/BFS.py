import os

from queue import Queue
import random


def BFS(kb, entity1, entity2):
    # Breadth First Search
    res = foundPaths(kb)
    res.markFound(entity1, None, None)
    q = Queue()
    q.put(entity1)
    while (not q.empty()):
        curNode = q.get()
        for path in kb.getPathsFrom(
                curNode):  # kb.getPathsFrom(curNode) is all the outgoing nodes of curNode, its type is List[Path]
            # Path class has two attributes: relation & connected_entity
            nextEntity = path.connected_entity
            connectRelation = path.relation
            if (not res.isFound(nextEntity)):  # not visited yet
                q.put(nextEntity)
                res.markFound(nextEntity, curNode, connectRelation)
            if (nextEntity == entity2):
                entity_list, path_list = res.reconstructPath(entity1, entity2)  # found a path
                return (True, entity_list, path_list)
    return (False, None, None)  # not found a path yet


def test():
    pass


class foundPaths(object):
    def __init__(self, kb):
        self.entities = {}
        for entity, relations in kb.entities.items():
            self.entities[entity] = (False, "", "")

    def isFound(self, entity):
        return self.entities[entity][0]  # True or False

    def markFound(self, entity, prevNode, relation):
        self.entities[entity] = (True, prevNode, relation)  # Lable the proceeding entities and relations

    def reconstructPath(self, entity1, entity2):  # Reconstruct the reasoning paths
        entity_list = []  # Save entities
        path_list = []  # Save relations
        curNode = entity2
        while (curNode != entity1):
            entity_list.append(curNode)

            path_list.append(self.entities[curNode][2])  # Relation
            curNode = self.entities[curNode][1]  # Walk back to the proceeding entity
        entity_list.append(curNode)  # Entity
        entity_list.reverse()
        path_list.reverse()
        return (entity_list, path_list)

    def __str__(self):
        res = ""
        for entity, status in self.entities.items():
            res += entity + "[{},{},{}]".format(status[0], status[1], status[2])
        return res

    __repr__ = __str__
