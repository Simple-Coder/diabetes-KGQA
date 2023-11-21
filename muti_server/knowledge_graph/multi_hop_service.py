"""
Created by xiedong
@Date: 2023/11/21 17:45
"""
all_path_str = [(['糖尿病', '二甲双胍', '二甲双胍', '糖尿病'], ['治疗药物', 'Equal', 'inv_治疗药物']),
                (['鱼', '鱼', '鱼', '鱼'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '二甲双胍', '二甲双胍', '糖尿病'], ['治疗药物', 'Equal', 'inv_治疗药物']),
                (['鸡蛋', '鸡蛋', '鸡蛋', '鸡蛋'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['白糖', '甜叶菊', '甜叶菊', '甜叶菊'], ['替代品', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['红糖', '红糖', '红糖', '红糖'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['多尿', '多尿', '多尿', '多尿'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['多食', '多食', '多食', '多食'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['肾盂肾炎', '肾盂肾炎', '肾盂肾炎', '肾盂肾炎'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['膀胱炎', '膀胱炎', '膀胱炎', '膀胱炎'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['脓毒血症', '脓毒血症', '脓毒血症', '脓毒血症'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '尿糖', '尿糖', '尿糖'], ['疾病检查', 'Equal', 'Equal']),
                (['血糖', '血糖', '血糖', '血糖'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '血糖', '血糖', '血糖'], ['疾病检查', 'Equal', 'Equal']),
                (['尿糖', '尿糖', '尿糖', '尿糖'], ['Equal', 'Equal', 'Equal']),
                (['糖尿病', '格列苯脲', '默沙东', '二甲双胍'], ['治疗药物', '生产厂商', 'inv_生产厂商']),
                (['二甲双胍', '默沙东', '二甲双胍', '默沙东'], ['生产厂商', 'inv_生产厂商', '生产厂商']),
                (['糖尿病', '二甲双胍', '默沙东', '二甲双胍'], ['治疗药物', '生产厂商', 'inv_生产厂商']),
                (['格列苯脲', '默沙东', '二甲双胍', '糖尿病'], ['生产厂商', 'inv_生产厂商', 'inv_治疗药物']),
                (['糖尿病', '二甲双胍', '默沙东', '二甲双胍'], ['治疗药物', '生产厂商', 'inv_生产厂商']),
                (['罗格列酮', '默沙东', '二甲双胍', '糖尿病'], ['生产厂商', 'inv_生产厂商', 'inv_治疗药物']),
                (['二甲双胍', '糖尿病', '二甲双胍', '都乐宁'], ['inv_治疗药物', '治疗药物', '生产厂商']),
                (['默沙东', '格列苯脲', '糖尿病', '二甲双胍'], ['inv_生产厂商', 'inv_治疗药物', '治疗药物']),
                (['二甲双胍', '糖尿病', '二甲双胍', '默沙东'], ['inv_治疗药物', '治疗药物', '生产厂商']),
                (['都乐宁', '都乐宁', '都乐宁', '都乐宁'], ['Equal', 'Equal', 'Equal']),
                (['格列苯脲', '糖尿病', '二甲双胍', '默沙东'], ['inv_治疗药物', '治疗药物', '生产厂商']),
                (['默沙东', '二甲双胍', '糖尿病', '二甲双胍'], ['inv_生产厂商', 'inv_治疗药物', '治疗药物']),
                (['罗格列酮', '糖尿病', '二甲双胍', '默沙东'], ['inv_治疗药物', '治疗药物', '生产厂商']),
                (['默沙东', '二甲双胍', '糖尿病', '二甲双胍'], ['inv_生产厂商', 'inv_治疗药物', '治疗药物']),
                (['白糖', '糖尿病', '白糖', '白糖'], ['inv_忌吃食物', '忌吃食物', 'Equal']),
                (['甜叶菊', '甜叶菊', '甜叶菊', '甜叶菊'], ['Equal', 'Equal', 'Equal'])]


def print_paths(all_path):
    path_set = set()
    for idx, element in enumerate(all_path):
        entities = element[0]
        relations = element[1]

        for i, entity in enumerate(entities):
            if i > 0 and i <= len(relations):
                rel = relations[i - 1]

                head = entities[i - 1]
                tail = entities[i]

                if head == tail:
                    continue

                path = head + ',' + rel + '->' + tail

                history_paths = is_path_exists(path_set, head)
                if history_paths and rel.find('inv') < 0:
                    for path in history_paths:
                        path += ',' + rel + '->' + tail
                path_set.add(path)
    return path_set


def is_path_exists(path_set, head):
    matching_paths = []
    for path in path_set:
        if path.endswith(head) and path.find("inv_") < 0:
            # if path.endswith(head):
            matching_paths.append(path)
    return matching_paths


def print_paths1(path):
    current_path = []
    count_total = 0
    for element in path:
        count_total += 1
        # if element != 'Equal' and not count_total % 4 == 0:
        if element != 'Equal':
            current_path.append(element)
            # current_path.append('->')
        else:
            if current_path:
                print("->".join(current_path))
                # print(current_path)
            current_path = []
    if current_path:
        print("->".join(current_path))


def collect_result(paths):
    result = []
    for i, pathi in enumerate(paths):
        head = pathi.split(',')[0]
        usePathi = True
        for j, pathj in enumerate(paths):
            splitj = pathj.split('->')
            headj_relj = splitj[0]
            tail = splitj[-1]
            if head == tail and pathj.find('inv') < 0 and pathi.find('inv') < 0:
                print("find")
                headj_relj += '->' + pathi
                usePathi = False
                result.append(headj_relj)

        if usePathi:
            result.append(pathi)

    for ret in result:
        print(ret)

    return result


paths_sets = print_paths(all_path_str)
result = collect_result(paths_sets)


# paths_sets = print_paths(paths_sets)
# for path in paths_sets:
#     print(path)


def find_answer(head, relation, jump_num):
    filtered_list = [item for item in result if item.count("->") == jump_num]

    answer = []

    for path in filtered_list:
        path_elements = path.split('->')
        elements_one = path_elements[1]
        split_e_ones = elements_one.split(',')
        if split_e_ones[1] == relation:
            answer.append(path)

    return answer
    print(answer)


def convert_answer(head, reltion, template, answer):
    if len(answer) == 0:
        print('不知道如何回答~')

    group_path = {}
    for ans in answer:
        path_elements = ans.split('->')
        e1 = path_elements[1]
        e1_split = e1.split(',')
        e1 = e1_split[0]
        e2 = path_elements[2]

        if e2 not in group_path:
            group_path[e2] = []

        group_path[e2].append(e1)

    # print(group_path)
    # 创建一个空字符串，用于存储描述信息
    description = template

    # 遍历字典中的键值对，生成描述信息
    for head, relations in group_path.items():
        # 生成描述信息的一部分，包括 head 和对应的 relations
        relation_description = f"{head}({reltion})："

        # 添加每个关系到描述信息中
        relation_description += "、".join(relations)

        # 将该部分描述信息添加到总的描述信息中
        description += relation_description
        description += "\n"

    # 打印生成的自然语言描述
    print('----')
    print('生成回答：\n')
    print(description)


answer = find_answer('糖尿病', '生产厂商', 2)
convert_answer('糖尿病', '生产厂商', '糖尿病治疗药物的生产厂商有:\n', answer)
print('---')
answer = find_answer('糖尿病', '替代品', 2)
convert_answer('糖尿病', '替代品', '糖尿病忌吃食物的替代品有:\n', answer)
# find_answer('糖尿病', 'aa', 2)
print()
