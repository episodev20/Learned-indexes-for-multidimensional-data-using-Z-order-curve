import sys
sys.modules[__name__].__dict__.clear()

import pandas
import morton
import time
import copy
import os
import numpy as np
from pandas import DataFrame
import seaborn as sns
from sklearn import preprocessing

df = pandas.DataFrame(index=range(0,100), columns=range(0,100))
m = morton.Morton(dimensions=2, bits=32)
for x in range(0, len(df)):
    for y in range(0, len(df)):
        df[x][y] = m.pack(x, y)



x = df.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pandas.DataFrame(x_scaled)


Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
Cols = ['A', 'B', 'C', 'D']
df = DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)

df.style.background_gradient(cmap='Blues')