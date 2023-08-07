"""
Created by xiedong
@Date: 2023/8/7 12:50
"""
# nlu.py
"""
jieba_tags 


n：普通名词
nr：人名
ns：地名
nt：机构名
nz：其他专名
nl：名词性惯用语
ng：名词性语素
s：处所词
t：时间词
f：方位词
v：动词
vd：副动词
vn：名动词
a：形容词
ad：副形词
an：名形词
d：副词
m：数量词
q：量词
r：代词
p：介词
c：连词
u：助词
xc：其他虚词
w：标点符号
PER：人名
LOC：地名
ORG：机构名
"""

import jieba.posseg as pseg


def query_understanding(query):
    # 分词和词性标注
    words = pseg.cut(query)
    word_tags = [(word, tag) for word, tag in words]

    main_entity = None
    domain = None
    question_type = None

    # 实体识别
    for word, tag in word_tags:
        if tag == 'ns':  # 假设 'ns' 标记表示地名实体
            main_entity = word
            break

    # 业务领域识别
    if '旅游' in query:
        domain = '旅游'
    elif '健康' in query:
        domain = '健康'
    # ... 根据具体需求继续添加其他领域判断

    # 问题类型识别
    if '怎么去' in query:
        question_type = '导航'
    elif '门票' in query:
        question_type = '一跳'
    # ... 根据具体需求继续添加其他问题类型判断

    return main_entity, domain, question_type


if __name__ == '__main__':
    main_entity, domain, question_type = query_understanding('故宫周末有门票吗')
    print(main_entity)
    print(domain)
    print(question_type)
