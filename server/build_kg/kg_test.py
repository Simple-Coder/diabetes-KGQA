"""
Created by xiedong
@Date: 2023/5/23 12:29
"""
from py2neo import Graph


class TestTractir():
    def __init__(self):
        self.graph = Graph(
            host="127.0.0.1",
            http_port=7474,
            user="neo4j",
            password="123456")

    def extract_triples(self, label):
        data = []
        links = []
        gql = f"MATCH(p:`临床表现`)-[r:Symptom_Disease]->(q:`疾病`) where q.name='{label}' RETURN type(r) AS type,id(r) as Relid,p,q"

        kgdata = self.graph.run(gql).data()

        count = 0
        for value in kgdata:
            count += 1
            relNode = value['type']
            Relid = value['Relid']
            pNode = value['p']
            qNode = value['q']
            if count == 1:
                data.append({'id': str(qNode.identity), 'name': qNode['name'], 'des': qNode['name']})
            else:
                data.append({'id': str(pNode.identity), 'name': pNode['name'], 'des': pNode['name']})
            links.append(
                {'source': str(qNode.identity), 'target': str(pNode.identity), 'value': relNode,
                 'id': str(Relid)})

        return [data, links]


if __name__ == '__main__':
    tractir = TestTractir()
    triples = tractir.extract_triples('糖尿病')

    print()
