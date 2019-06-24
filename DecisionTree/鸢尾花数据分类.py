# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=FutureWarning)

iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

path = "./datas/iris.data"
data = pd.read_csv(path, header=None)
x = data[list(range(4))]  # 获取X变量

y = pd.Categorical(data[4]).codes  # 把Y转换成分类型的0,1,2
print("总样本数目 ：%d,特征属性数目：%d" % x.shape)
# print(x.head())
# print(y)

#数据进行分割(训练数据和测试数据)
x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y,train_size=0.8,random_state=14)

x_train,x_test,y_train,y_test = x_train1,x_test1,y_train1,y_test1
print("训练数据集样本数目：%d,测试数据集样本数目:%d "%(x_train.shape[0],x_test.shape[0]))

#数据归一化
ss = MinMaxScaler()
x_train  =ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print("原始数据各个特征属性的调整最小值：",ss.min_)
print("原始数据各个特征属性的缩放数据值：",ss.scale_)

#特征选择
ch2 = SelectKBest(chi2,k=3)
x_train = ch2.fit_transform(x_train, y_train)
x_test = ch2.transform(x_test)

select_name_index =ch2.get_support(indices=True)
print(select_name_index)
print("对类别判断影响最大的三个特征属性分别是：",ch2.get_support(indices=False))

#降维
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#模型的构建
model = DecisionTreeClassifier(criterion="entropy",random_state=0)
model.fit(x_train,y_train)
#模型预测
y_test_hat = model.predict(x_test)

y_test2 = y_test.reshape(-1)
result = (y_test2==y_test_hat)
print("准确率：%.2f%%"%(np.mean(result)*100))
#实际可通过参数获取
print("Scores:",model.score(x_test,y_test))
print("Classes:",model.classes_)
print("获取各个特征的权重：",model.feature_importances_)

#画图
N = 100
x1_min = np.min((x_train.T[0].min(),x_test.T[0].min()))
x1_max  = np.max((x_train.T[0].max(),x_test.T[0].max()))
x2_min = np.min((x_train.T[1].min(),x_test.T[1].min()))
x2_max = np.max((x_train.T[1].max(),x_test.T[1].max()))

t1 = np.linspace(x1_min,x1_max,N)
t2 = np.linspace(x2_min,x2_max,N)
x1,x2 = np.meshgrid(t1,t2)
x_show = np.dstack((x1.flat,x2.flat))[0]

y_show_hat = model.predict(x_show)

y_show_hat = y_show_hat.reshape(x1.shape)

plt_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
plt_dark = mpl.colors.ListedColormap(['g','r','b'])

plt.figure(facecolor='w')
plt.pcolormesh(x1,x2,y_show_hat,cmap=plt_light)
#画测试数据的点信息
plt.scatter(x_test.T[0],x_test.T[1],c=y_test.ravel(),edgecolors='k',s=150,zorder=10,cmap=plt_dark,marker='*')
#画训练数据的点信息
plt.scatter(x_train.T[0],x_train.T[1],c=y_train.ravel(),edgecolors='k',s=40,cmap=plt_dark)
plt.xlabel(u'特征属性1',fontsize=15)
plt.ylabel(u'特征属性2',fontsize=15)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.grid(True)
plt.title(u"鸢尾花数据的决策树分类",fontsize=18)
plt.show()

"""参数优化"""
pipe = Pipeline([
    ('mms',MinMaxScaler()),
    ('skb',SelectKBest(chi2)),
    ('pca',PCA()),
    ('decision',DecisionTreeClassifier(random_state=0))
])

parameters = {
    'skb__k':[1,2,3,4],
    'pca__n_components':[0.5,0.99],
    'decision__criterion':['gini','entropy'],
    'decision__max_depth':[1,2,3,4,5,6,7,8,9,10]
}

x_train2,x_test2,y_train2,y_test2 = x_train1,x_test1,y_train1,y_test1

gscv = GridSearchCV(pipe,param_grid=parameters,cv=3)
gscv.fit(x_train2,y_train2)
print("最优参数列表：",gscv.best_params_)
print("score值：",gscv.best_score_)
print("最优模型：",gscv.best_estimator_)
# 预测值
y_test_hat2 = gscv.predict(x_test2)

### 应用最优参数看效果
mms_best = MinMaxScaler()
skb_best = SelectKBest(chi2,k=3)
pca_best = PCA(n_components=0.99)
decision3  = DecisionTreeClassifier(criterion='gini',max_depth=4)
#构建模型并训练模型
x_train3,x_test3,y_train3,y_test3 = x_train1,x_test1,y_train1,y_test1
x_train3 = pca_best.fit_transform(skb_best.fit_transform(mms_best.fit_transform(x_train3), y_train3))
x_test3 = pca_best.transform(skb_best.transform(mms_best.transform(x_test3)))

decision3.fit(x_train3,y_train3)

print("正确率：",decision3.score(x_test3,y_test3))


### 基于原始数据前3列比较一下决策树在不同深度的情况下的错误率
x_train4,x_test4,y_train4,y_test4 = train_test_split(x.iloc[:,:2],y,train_size=0.7,random_state=14)
depths = np.arange(1,15)
err_list = []
train_err_list =[]
for d in depths:
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=d,min_samples_split=10)
    clf.fit(x_train4,y_train4)

    score = clf.score(x_test4,y_test4)
    err = 1-score
    err_list.append(err)
    train_err_list.append(1-clf.score(x_train4,y_train4))
    print("%d深度，训练集上正确率%.5f"%(d,clf.score(x_train4,y_train4)))
    print("%d深度，测试集上正确率%.5f\n"%(d,score))

plt.figure(facecolor='w')
plt.plot(depths,err_list,'ro--',lw=3)
plt.plot(depths,train_err_list,'b*--',lw=3)
plt.xlabel(u"决策树深度",fontsize=16)
plt.ylabel(u'错误率',fontsize=16)
plt.grid(True)
plt.title(u"决策树层次太多导致的拟合问题(欠拟合和过拟合)",fontsize=18)
plt.show()
