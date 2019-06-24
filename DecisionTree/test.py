# -*- coding: utf-8 -*-
import math

w1 = math.exp(-0.6112 * 1 * 1)
w2 = math.exp(-0.6112 * 1 * (-1))
z = 0.1 * w1 * 7 + 0.1 * w2 * 3
print(z)
# print(0.1*w1*7/z+0.1*w2*3/z)
print(0.1 * w1 / z)

import xgboost as xgb

