"""
Created by xiedong
@Date: 2023/4/5 21:46
"""
import requests
import json


def crawl_wenzhen_data():
    url = 'https://www.haodf.com/citiao/jibing-tangniaobing/bingcheng.html'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36'
    }

    response = requests.get(url, headers=headers)

    try:
        data = response.json()
        wenzhen_data = []
        for item in data['data']['questions']:
            question = item['question']
            answer = item['answer']
            wenzhen_data.append({'question': question, 'answer': answer})

        with open('wenzhen_data.json', 'w', encoding='utf-8') as file:
            json.dump(wenzhen_data, file, ensure_ascii=False)

        print("数据爬取完成并保存成功！")

    except ValueError:
        print("无法解析JSON响应:")
        print(response.text)


if __name__ == '__main__':
    crawl_wenzhen_data()
