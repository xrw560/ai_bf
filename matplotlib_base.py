# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

# print(plt.rcParams.keys())
# plt.rcdefaults()
# t = np.arange(0.0, 1.0, 0.01)
# s = np.sin(2 * np.pi * t)
# plt.rcParams['lines.color'] = 'r'
# plt.plot(t, s)
#
# c = np.cos(2 * np.pi * t)
# plt.rcParams['lines.linewidth'] = '3'
# plt.plot(t, c)
# plt.show()

# x1 = [1, 2, 3]
# y1 = [5, 7, 4]
# x2 = [1, 2, 3]
# y2 = [10, 14, 12]
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.plot(x1, y1, 'ro--', label='First Line')
# plt.plot(x2, y2, 'b--', label='Second Line')
# plt.xlabel('月份')
# plt.ylabel('年份')
# plt.title('进出口数据')
# plt.xlim(1, 3)
# plt.ylim(0, 15)
# plt.xticks(np.linspace(0, 6, 5))
# plt.yticks(np.arange(1, 15, 3), ['2011年', '2012年', '2013年', '2014年', '2015年'])
# ax = plt.gca()
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')
# plt.legend()
# plt.show()


# mpl.rcParams['font.sans-serif']=['SimHei']
# mpl.rcParams['axes.unicode_minus']=False
#
# plt.figure()
# plt.bar([1,3,5,7,9,11],[5,2,7,8,2,6],label='Example one',color='y')
# plt.bar([2,4,6,8,10,12],[8,6,2,5,6,3],label='Example two',color='g')
# plt.legend()
# plt.xlabel('bar 序号')
# plt.ylabel('bar 高度')
# plt.title('test')
# plt.show()


s = pd.Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
print(s)
s.plot()
plt.show()
