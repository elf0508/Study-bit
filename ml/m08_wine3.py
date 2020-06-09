import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기

wine = pd.read_csv('./data/csv/winequality-white.csv',
                   header = 0, index_col = None,
                   sep = ';', encoding = 'cp949')

count_data = wine.groupby('quality')['quality'].count()  # 컬럼내에 있는 개체들을 행별로 갯수를 세겠다.

print(count_data)

'''
quality  
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
'''

# 시각화

count_data.plot()
plt.show()


