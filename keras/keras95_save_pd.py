import numpy as np

import pandas as pd

# 판다스로 불러들인다.
datasets = pd.read_csv("./data/csv/iris.csv", index_col = None,
                        header = 0, sep =',')

print(datasets)
'''
     150    4  setosa  versicolor  virginica
0    5.1  3.5     1.4         0.2          0 <-- 실제 데이터 인식부분(시작)
1    4.9  3.0     1.4         0.2          0
2    4.7  3.2     1.3         0.2          0
3    4.6  3.1     1.5         0.2          0
..   ...  ...     ...         ...        ...
145  6.7  3.0     5.2         2.3          2
146  6.3  2.5     5.0         1.9          2
147  6.5  3.0     5.2         2.0          2
148  6.2  3.4     5.4         2.3          2
149  5.9  3.0     5.1         1.8          2 <-- 실제 데이터 인식부분(끝)

[150 rows x 5 columns]
'''

print(datasets.head())  # 위에서부터 5개만

print(datasets.tail())  # 아래에서부터 5개만

print("====판다스를 넘파이로 바꾸는 함수(.values)========")
print(datasets.values)

aaa = datasets.values
print(type(aaa))  # <class 'numpy.ndarray'>

# 넘파이로 저장하시오.

# 저장

np.save('./data/iris_np_save.npy', arr=aaa)     