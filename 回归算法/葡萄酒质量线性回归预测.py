# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn import metrics

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

# 读取数据
path1 = "datas/winequality-red.csv"
path2 = "datas/winequality-white.csv"

df1 = pd.read_csv(path1, sep=";")
df1['type'] = 1
df2 = pd.read_csv(path2, sep=";")
df2['type'] = 2
df = pd.concat([df1, df2], axis=0)
# print(df.head())
names = ["fixed acidity", "volatile acidity", "citric acid",
         "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates",
         "alcohol", "type"]
quality = "quality"
# print(df.info())

names1 = []
for i in list(df):
    names1.append(i)
# print(names1)

# 异常数据处理
new_df = df.replace("?", np.nan)
datas = new_df.dropna(how="any")  # 只要有列为空，就进行行删除操作

X = datas[names]
Y = datas[quality]
# print(Y.ravel())


# 创建模型列表
models = [
    Pipeline([
        ('ss', StandardScaler()),
        ("Poly", PolynomialFeatures()),  # 多项式
        ("linear", LinearRegression())
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('Poly', PolynomialFeatures()),
        ('linear', RidgeCV(alphas=np.logspace(-4, 2, 20)))
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('Poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-4, 2, 20)))
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('Poly', PolynomialFeatures()),
        ('linear', ElasticNetCV(alphas=np.logspace(-4, 2, 20), l1_ratio=np.logspace(0, 1, 5)))
    ])
]

## 将数据集划分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=0)
len_x_test = range(len(X_test))
titles = '线性回归预测', "Ridge", "Lasso", "ElasticNet"
d_pool = np.arange(1, 4, 1)  # 1 2 3阶多项式扩展
m = len(d_pool)
clrs = ['green', 'black', 'yellow', 'blue']
for t in range(4):
    plt.subplot(2, 2, t + 1)
    model = models[t]
    plt.plot(len_x_test, Y_test, c='r', lw=2, label="真实值")
    for i, d in enumerate(d_pool):
        # 设置阶数
        model.set_params(Poly__degree=d)
        # 模型训练
        model.fit(X_train, Y_train)
        # 预测
        Y_pre = model.predict(X_test)
        R = model.score(X_test, Y_test)

        # 输出信息
        lin = model.get_params()['linear']
        output = "%s:%d阶，截距:%d，系数：" % (titles[t], d, lin.intercept_)
        print(output, lin.coef_)
        plt.plot(len_x_test, Y_pre, c=clrs[i], lw=2, label="%d阶预测值，$R^2$=%.3f" % (d, R))
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.title(titles[t], fontsize=18)
    plt.xlabel('X', fontsize=18)
    plt.ylabel('Y', fontsize=18)
plt.suptitle('葡萄酒质量预测', fontsize=22)
plt.show()
