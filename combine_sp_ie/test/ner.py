"""
Created by xiedong
@Date: 2023/8/7 15:40
"""
from ltp import LTP

ltp = LTP()

result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "ner"])
print(result.ner)
