# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

df01 = DataFrame([['Tom', 'Gerry', 'John'], [76, 98, 85]])
df01.columns=['1','2','3']
df01['new']=df01['1']+df01['2']
print(df01)
#
# df02 = DataFrame([
#     ['Tom', 76],
#     ['Gerry', 98],
#     ['John', 85]
# ], columns=['姓名', '成绩'])
# print(df02)
# print(df02.index)
# print(df02.values)

# arr = np.array([
#     ['Tom',76],
#     ['Gerry',98],
#     ['John',85]
# ])
# df03 = DataFrame(arr,index=['one','two','three'],columns=['name','score'])
#
# print(df03)

# # 通过字典创建
# dic = {
#     '语文': [70, 80, 90],
#     '数学': [80, 88, 90],
#     '英语': [80, 78, 88]
# }
# df02 = pd.DataFrame(dic)
# # print(df02)
#
# df02.index = ['one', 'two', 'three']  # 重置索引
# # print(df02)
#
# # df02.columns=list('abc')
# df03 = pd.DataFrame(np.random.randint(1, 9, (3, 3)), index=list("ABC"), columns=list("abc"))
# print(df03)
# datas = {
#     'name': ['aaa', 'bbb', 'ccc'],
#     'age': [70, 80, 90],
#     '语文': [70, 80, 88],
#     '数学': [82, 78, 80],
#     '英语': [86, 88, 90]
# }
# df = pd.DataFrame(datas)
# datas = {
# 	'name':['aaa','bbb','ccc'],
# 	'age':[70,80,np.nan],
# 	'语文':[70,np.nan,np.nan],
# 	'数学':[np.nan,78,80],
# 	'英语':[86,88,90]
# }
# df = pd.DataFrame(datas)
# print(df.notnull())

# data = pd.Series([100, 200, 122, 150, 180], index=[
#     ['2016', '2016', '2016', '2017', '2017'],
#     ['苹果', '香蕉', '西瓜', '苹果', '西瓜']
# ])
# print(data)
# # data01 = data.swaplevel().sort_index()
# # print(data01)
# data02 = data.unstack(level=1)
# print(data02.stack(level=0))
