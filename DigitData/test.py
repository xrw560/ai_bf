# -*- coding: utf-8 -*-
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import seaborn

# 采样点选择1400个，因为设置的信号频率分量最高为600Hz，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400Hz（即一秒内有1400个采样点）
x = np.linspace(0, 1, 1400)

# 设置需要采样的信号，频率分量有180，390和600
y = 7 * np.sin(2 * np.pi * 180 * x) + 1.5 * np.sin(2 * np.pi * 390 * x) + 5.1 * np.sin(2 * np.pi * 600 * x)

yy = fft(y)  # 快速傅里叶变换
yreal = yy.real  # 获取实数部分
yimag = yy.imag  # 获取虚数部分

yf = abs(fft(y))  # 取模
yf1 = abs(fft(y)) / ((len(x) / 2))  # 归一化处理
yf2 = yf1[range(int(len(x) / 2))]  # 由于对称性，只取一半区间

xf = np.arange(len(y))  # 频率
xf1 = xf
xf2 = xf[range(int(len(x) / 2))]  # 取一半区间

# 原始波形
plt.subplot(221)
plt.plot(x[0:50], y[0:50])
plt.title('Original wave')
# 混合波的FFT（双边频率范围）
plt.subplot(222)
plt.plot(xf, yf, 'r')  # 显示原始信号的FFT模值
plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
# 混合波的FFT（归一化）
plt.subplot(223)
plt.plot(xf1, yf1, 'g')
plt.title('FFT of Mixed wave(normalization)', fontsize=9, color='r')

plt.subplot(224)
plt.plot(xf2, yf2, 'b')
plt.title('FFT of Mixed wave)', fontsize=10, color='#F08080')

plt.show()
