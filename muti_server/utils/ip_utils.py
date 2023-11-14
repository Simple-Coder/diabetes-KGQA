"""
Created by xiedong
@Date: 2023/11/14 16:30
"""
from xdbSearcher import XdbSearcher

import atexit


class IP2RegionSearcher:
    def __init__(self, db_path):
        # 预先加载整个 xdb
        self.cb = XdbSearcher.loadContentFromFile(dbfile=db_path)

        # 仅需要使用上面的全文件缓存创建查询对象, 不需要传源 xdb 文件
        self.searcher = XdbSearcher(contentBuff=self.cb)

        # 在程序退出时调用 self.close 方法
        atexit.register(self.close)

    def search(self, ip):
        # 执行查询
        region_str = self.searcher.search(ip)
        return region_str

    def close(self):
        # 关闭 searcher
        self.searcher.close()


# 在程序中创建 IP2RegionSearcher 对象
db_path = 'D://dev//PycharmProjects//diabetes-KGQA//muti_server//utils//ip//data//ip2region.xdb'
ip_searcher = IP2RegionSearcher(db_path)

# 使用查询方法
# ip_result = ip_searcher.search("114.114.114.114")
ip_result = ip_searcher.search("127.0.0.1")
print(ip_result)

# 程序退出时会自动调用 ip_searcher.close()，释放资源
