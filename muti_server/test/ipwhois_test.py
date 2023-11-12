"""
Created by xiedong
@Date: 2023/11/12 22:08
"""
import requests
import json

import requests


def get_binary_ip(ip):
    """
    将IP地址转为二进制
    :param ip: IP地址，例如："192.168.1.1"
    :return: IP地址的二进制表示，例如："11000000.10101000.00000001.00000001"
    """
    binary_ip = ".".join([bin(int(x) + 256)[3:] for x in ip.split(".")])
    return binary_ip


def get_location_by_ip(ip):
    """
    通过HTTP请求查询IP地址归属地
    :param ip: IP地址，例如："192.168.1.1"
    :return: IP地址的归属地，例如："广东省深圳市"
    """
    url = "http://ip.taobao.com/service/getIpInfo.php?ip=" + ip
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data["code"] == 0:
            location = data["data"]["region"] + data["data"]["city"]
        else:
            location = "未知"
    except Exception as e:
        print(e)
        location = "未知"
    return location


if __name__ == "__main__":
    ip = "192.168.1.1"
    binary_ip = get_binary_ip(ip)
    location = get_location_by_ip(ip)
    print("IP地址:{}的归属地是:{}".format(ip, location))
