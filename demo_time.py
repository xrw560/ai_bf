# -*- coding: utf-8 -*-

import time
import datetime
import locale

locale.setlocale(locale.LC_CTYPE, "chinese")

# 获取当前时间 时间戳
now_time = time.time()

now_tuple = time.localtime(now_time)

# print(time.asctime(now_tuple))

# print(time.strftime("%Y-%m-%d %H:%M:%S", now_tuple))

# 自定义显示日期格式
now_date = time.strftime("%Y-%m-%d %H:%M:%S", now_tuple)
# 注意，如果直接strftime写中文的时候，编码会解析失败，设置解析为中文
print(time.strftime("%Y年%m月%d %H:%M:%S", now_tuple))
# 把可读时间格式转换为时间元组
date_tuple = time.strptime(now_date, "%Y-%m-%d %H:%M:%S")
# 把时间元组转换为时间戳
print(time.mktime(date_tuple))

# # clock 获取进程时间
# print(time.clock())  # ----开始 从 0计数 ----
# for i in range(1000000):
#     pass
# one_time = time.clock()  # 0.03671786919984463
# print(one_time)
# for i in range(1000000):
#     pass
# print(time.clock() - one_time)  # 0.06938751180321552

# # sleep(n) 进程暂停n秒
# for i in range(4):
#     print(i)
#     time.sleep(2)
