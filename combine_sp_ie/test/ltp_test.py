"""
Created by xiedong
@Date: 2023/8/7 12:46
"""
"""
pip install ltp
"""
from ltp import LTP

ltp = LTP()

words = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws"], return_dict=False)
# [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
print(words)
result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos"])
print(result)
print(result.pos)
# [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
# [['r', 'v', 'nh', 'v', 'v', 'n', 'wp']]
