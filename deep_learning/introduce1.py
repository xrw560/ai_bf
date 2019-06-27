# -*- coding: utf-8 -*-
# x = 1
# a = 2
# y = lambda x: x ** 2
#
# dy = lambda x, a: a - y(x)
#
# y_x = lambda x: 2 * x  # 导数
#
# dx = lambda x, a: dy(x, a) / y_x(x)
#
# for _ in range(5):
#     x += dx(x, a)
#
# # print(x)
#
# def mysqrt(a):
#     x = 1
#     for _ in range(100):
#         x+= dx(x,a)
#     return x


# print(mysqrt(3))
# y = lambda x:x**3
# dy = lambda x,a:a-y(x)
# y_x = lambda x: 3*x**2
# dx = lambda x,a:dy(x,a)/y_x(x)
# x=1
# a = 2
# for _ in range(100):
#     x += dx(x,a)
#
# print(x)


# y = lambda x,a: (x**2-a)**2
# y_x = lambda x,a:4*x*(x**2-a)
# dx = lambda x,a,lr:-y_x(x,a)*lr
# def mysqrt(a,repeats,lr):
#     x=1
#     for _ in range(repeats):
#         x+=dx(x,a,lr)
#     return x
# print(mysqrt(2,10000,0.01))


# y = lambda x1, x2: (x1 - 3) ** 2 + (2 * x2 + 4) ** 2
# y_x1 = lambda x1: 2 * (x1 - 3)
# y_x2 = lambda x2: 4 * (2 * x2 + 4)
#
# dx1 = lambda x1, lr: -y_x1(x1) * lr
# dx2 = lambda x2, lr: -y_x2(x2) * lr
#
#
# def myfun(lr):
#     x1, x2 = 1, 1
#     for _ in range(10000):
#         x1 += dx1(x1, lr)
#         x2 += dx2(x2, lr)
#     return x1, x2


# print(myfun(0.001))
