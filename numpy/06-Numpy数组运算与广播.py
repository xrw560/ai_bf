# -*- coding:utf-8 -*-

import numpy as np


'''
    Numpy算数运算
    用于执行算数运算的输入数组必须具有相同的行列或符合数组广播规则

a = np.arange(9).reshape(3,3)
b = np.array([2,3,4])
print(a)
print(b)
print('**************************************')
#两个数组相加
print(np.add(a,b))
print(a+b)
print('**************************************')
#两个数组相减
print(a-b)
print(np.subtract(a,b))
print('**************************************')
#两个数组相乘
print(a*b)
print(np.multiply(a,b))
print('**************************************')
print(a/b)
print(np.divide(a,b))
print('**************************************')

#两个数组取幂
print(a**b)
print(np.power(a,b))
print('**************************************')
print()

print()
print('****************两个数组取余**********************')
#mod 两个数组取余
a = np.array([10,20,30])
b = np.array([3,5,7])
print(np.mod(a,b))
print(np.remainder(a,b))
print('**************************************')
print()
'''
'''
    数组的矩阵积
    两个二位矩阵，满足第一个矩阵的列数与第二个矩阵的行数相同，那么就可以进行矩阵的乘法，即矩阵积
'''
print()
arr1 = np.array([
    [110,60,220],
    [115,45,180],
    [132,67,209]
])
arr2 = np.array([
    [12.3,0.04],
    [204,2.34],
    [9.98,0.45]
])
print(arr2)
print(arr1.dot(arr2))
print('**************************************')








'''
    Numpy广播
    术语广播是指 NumPy 在算术运算期间处理不同形状的数组的能力。
    对数组的算术运算通常在相应的元素上进行，如果两个阵列具有完全相同的形状，则这些操作无缝执行
'''

a = np.array([1,2,3,4])
b = np.array([[10,20,30,40],[1,2,3,4]])
c = a * b
print(c)

'''
    如果两个数组的维数不相同，则元素到元素的操作是不可能的。 
    然而，在 NumPy 中仍然可以对形状不相似的数组进行操作，因为它拥有广播功能。
    较小的数组会广播到较大数组的大小，以便使它们的形状可兼容。
    如果满足以下规则，可以进行广播：
        让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐
        输出数组的shape是输入数组shape的各个轴上的最大值
        如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，
        否则出错
        当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值
        
'''





