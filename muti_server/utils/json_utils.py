"""
Created by xiedong
@Date: 2023/6/10 10:05
"""

import jsonpickle


class NoTypeInfoEncoder(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        # 禁止打印类型信息，直接返回对象数据
        return obj


def json_str(obj):
    jsonpickle.handlers.registry.register(object, NoTypeInfoEncoder)
    # return jsonpickle.encode(obj)
    return jsonpickle.encode(obj, unpicklable=False).encode('utf-8')
