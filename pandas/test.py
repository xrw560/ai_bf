# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# df = pd.read_csv('data2.txt', sep=':', header=None)

df = pd.DataFrame(np.random.randint(1, 9, size=(4, 4)))
df.ix[1:2, 1] = np.NaN
df.ix[1:2, 2] = np.NaN
df.ix[1:2, 3] = np.NaN
df.ix[1, 0] = np.NaN
print(df)
# print(df.dropna())
# print(df.dropna(axis=1))

print(df.fillna({0: 0, 1: 10, 2: 20, 3: 30}))
