"""
Created by xiedong
@Date: 2023/5/28 20:41
"""

import os
import sys
import logging
from time import strftime

# 输出日志路径
PATH = os.path.abspath('D:\dev\PycharmProjects\diabetes-KGQA\combine_sp_ie') + '/logs/'
# PATH = os.path.abspath('/Users/xiedong/PycharmProjects/diabetes-KGQA/muti_server') + '/logs/'
# 设置日志格式#和时间格式
FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'


class MyLog(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.log_filename = '{0}{1}.log'.format(PATH, strftime("%Y-%m-%d"))

        # 清除已存在的处理程序，避免重复添加
        self.logger.handlers.clear()

        self.logger.addHandler(self.get_file_handler(self.log_filename))
        self.logger.addHandler(self.get_console_handler())
        # 设置日志的默认级别
        self.logger.setLevel(logging.DEBUG)

    # 输出到文件handler的函数定义
    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        return filehandler

    # 输出到控制台handler的函数定义
    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler


# 创建全局的MyLog实例
my_log = MyLog()

if __name__ == '__main__':
    PATH = os.path.abspath('D:\dev\PycharmProjects\diabetes-KGQA\muti_server') + '/logs/'
    print(PATH)