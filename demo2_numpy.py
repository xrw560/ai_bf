# -*- coding: utf-8 -*-
import numpy as np

# arr = np.array([[3, 9], [4, 6]])
# # print(np.tile(arr, 2))
#
# a = [[1, 2, 3], [4, 5, 6]]
# b = [[1, 2, 4], [3, 5, 6]]
# c = [[1, 3, 2], [5, 4, 6]]
# print(np.stack((a, b), axis=2))
#
# e = [1, 2, 3]
# f = [4, 5, 6]
# print((f,))
# print(np.stack((f,), axis=0 ))


# xarr = np.array([-1.1, -1.2, -1.3, -1.4, -1.5])
# yarr = np.array([-2.1, -2.2, -2.3, -2.4, -2.5])
# condition = xarr < yarr
# print(condition)
# print(np.where(condition, xarr, yarr))

arr = np.array([[1, 2, np.NAN, 4],
                [4, 5, 6, np.NAN],
                [7, 8, 9, np.NAN],
                [np.inf, np.e, np.pi, 4]])
condition = np.isnan(arr) | np.isinf(arr)
# print(condition)
print(np.where(condition, 0, arr))
