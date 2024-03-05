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

    start_docId = 10
    start_orgId = 100

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                medical_institution, doctor = line.split(' ', 1)
                doctor_id = start_docId  # 取医生名字哈希值的绝对值
                start_docId += 1
                org_id = start_orgId  # 取医疗结构名字哈希值的绝对值
                start_orgId += 1

                doctors[doctor_id] = doctor
                medical_institutions[org_id] = medical_institution
                doctor_to_org[doctor_id] = org_id

    return doctors, medical_institutions, doctor_to_org


def generate_doctors_json(doctors, doctor_to_org):
    doctors_json = []

    datalink = []
    datalink_id = 1000
    for doctor_id, doctor_name in doctors.items():
        doctor_info = {
            "category": 7,
            "des": doctor_name,
            "id": str(doctor_id),
            "name": doctor_name
        }
        doctors_json.append(doctor_info)

        link_info = {
            "id": datalink_id,
            "source": "1",
            "target": str(doctor_id),
            "value": "skilled_in"
        }
        datalink_id += 1
        datalink.append(link_info)
        link_info = {
            "id": datalink_id,
            "source": str(doctor_id),
            "target": str(doctor_to_org.get(doctor_id)),
            "value": "work_on"
        }
        datalink_id += 1
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

data = {
    'data': doctors_json + medical_institutions_json,
    'links': datalink
}
print("-----")
print(json.dumps(data, indent=2, ensure_ascii=False))
print("-----")
medical_institutions_str = "、".join(medical_institutions.values())
print("医疗结构名称列表：", medical_institutions_str)
