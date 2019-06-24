# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import *
from mpl_toolkits.mplot3d.axes3d import Axes3D

import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def f(x, y):
    return x ** 2 + y ** 2


def h(t):  # 导数
    return 2 * t


X = []
Y = []
Z = []

x = 2
y = 2
step = 0.1
f_change = x ** 2 + y ** 2
f_current = f(x, y)

X.append(x)
Y.append(y)
Z.append(f_current)

while f_change > 1e-10:
    x = x - step * h(x)
    y = y - step * h(y)
    f_change = f_current - f(x, y)  # delta
    f_current = f(x, y)
    X.append(x)
    Y.append(y)
    Z.append(f_current)

fig = plt.figure()
ax = Axes3D(fig)
X2 = np.arange(-2, 2, 0.2)
Y2 = np.arange(-2, 2, 0.2)
X2, Y2 = np.meshgrid(X2, Y2)
Z2 = X2 ** 2 + Y2 ** 2

ax.plot_surface(X2, Y2, Z2)
# ax.plot(X, Y, Z, 'ro--')
ax.set_title(u"梯度下降法求解，最终解为：x=%.2f,y=%.2f,z=%.2f" % (x, y, f_current))

# plt.show()
plt.savefig('fig.png', bbox_inches='tight')
