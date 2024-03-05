"""
Created by xiedong
@Date: 2024/3/5 12:34
"""

import json
import uuid


def read_data_from_file(filename):
    doctors = {}  # 医生字典，key为id，value为医生名字
    medical_institutions = {}  # 医疗结构字典，key为orgId，value为医疗结构名称
    doctor_to_org = {}  # 医生到医疗结构的映射，key为医生id，value为医疗结构id

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                medical_institution, doctor = line.split(' ', 1)
                doctor_id = abs(hash(doctor))  # 取医生名字哈希值的绝对值
                org_id = abs(hash(medical_institution))  # 取医疗结构名字哈希值的绝对值

                doctors[doctor_id] = doctor
                medical_institutions[org_id] = medical_institution
                doctor_to_org[doctor_id] = org_id

    return doctors, medical_institutions, doctor_to_org


def generate_doctors_json(doctors, doctor_to_org):
    doctors_json = []

    datalink = []
    for doctor_id, doctor_name in doctors.items():
        doctor_info = {
            "category": 7,
            "des": doctor_name,
            "id": str(doctor_id),
            "name": doctor_name
        }
        doctors_json.append(doctor_info)

        link_info = {
            "id": str(uuid.uuid4()),
            "source": "1",
            "target": str(doctor_id),
            "value": "skilled_in"
        }
        datalink.append(link_info)
        link_info = {
            "id": str(uuid.uuid4()),
            "source": str(doctor_id),
            "target": doctor_to_org.get(doctor_id),
            "value": "work_on"
        }
        datalink.append(link_info)
    doctors_json.append({
        "category": 2,
        "des": "2型糖尿病",
        "id": "1",
        "name": "2型糖尿病"
    })
    return doctors_json, datalink


def generate_medical_institutions_json(medical_institutions):
    medical_institutions_json = []
    for org_id, org_name in medical_institutions.items():
        institution_info = {
            "category": 8,
            "des": org_name,
            "id": str(org_id),
            "name": org_name
        }
        medical_institutions_json.append(institution_info)
    return medical_institutions_json


filename = "a.txt"  # 假设医生信息保存在名为doctors.txt的文件中
doctors, medical_institutions, doctor_to_org = read_data_from_file(filename)
doctors_json, datalink = generate_doctors_json(doctors, doctor_to_org)
medical_institutions_json = generate_medical_institutions_json(medical_institutions)

print("医生 JSON 数据：")
print(json.dumps(doctors_json, indent=2, ensure_ascii=False))
print("Link JSON 数据：")
print(json.dumps(datalink, indent=2, ensure_ascii=False))

print("\n医疗结构 JSON 数据：")
print(json.dumps(medical_institutions_json, indent=2, ensure_ascii=False))

data_data = []
data_data.append(doctors_json)
data_data.append(medical_institutions_json)
data = {
    'data': data_data,
    'link': datalink
}
print("-----")
print(json.dumps(data, indent=2, ensure_ascii=False))
print("-----")
medical_institutions_str = "、".join(medical_institutions.values())
print("医疗结构名称列表：", medical_institutions_str)
