# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# 绘制3维的散点图
# x = np.random.randint(0, 10, size=100)
# y = np.random.randint(-20, 20, size=100)
# z = np.random.randint(0, 30, size=100)

# 此处fig是二维
fig = plt.figure()

# 将二维转化为三维
axes3d = Axes3D(fig)
X2 = np.arange(-2, 2, 0.2)
Y2 = np.arange(-2, 2, 0.2)
X2, Y2 = np.meshgrid(X2, Y2)
Z2 = X2 ** 2 + Y2 ** 2
# axes3d.scatter3D(x,y,z)
# 效果相同
axes3d.plot_surface(X2, Y2, Z2)
plt.savefig('fig.png', bbox_inches='tight')
