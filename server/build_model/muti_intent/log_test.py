"""
Created by xiedong
@Date: 2023/5/28 18:30
"""
import logging

# 配置日志输出
logging.basicConfig(level=logging.INFO)

# 创建日志记录器
logger = logging.getLogger(__name__)

# 定义日志消息格式
log_format = "%(asctime)s - %(levelname)s - %(message)s"

# 配置日志格式
formatter = logging.Formatter(log_format)

# 创建日志处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 将日志处理器添加到记录器
logger.addHandler(console_handler)

# 打印带参数的日志
name = "John"
age = 30
logger.info("User %s is %d years old.", name, age)

