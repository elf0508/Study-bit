# 여러개의 컬럼 - 이상치의 위치

# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구하시오.
from pandas import Series, DataFrame
import numpy as np

def outliers(data_out):
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
    return outliers

a2 = np.array([[1, 5000],[200, 8],[2, 4],[3, 7],[8, 2]])
print(a2)
# [[   1 5000]
#  [ 200    8]
#  [   2    4]
#  [   3    7]
#  [   8    2]]

b2 = outliers(a2)
print(b2)
# [(array([1], dtype=int64),), (array([0], dtype=int64),)]

import pandas as pd

# pandas
def outliers(data_out):
        quartile_1 = data_out.quantile(.25)
        quartile_3 = data_out.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out > upper_bound) | (data_out < lower_bound))
         
a3 = pd.DataFrame({'a' : [1, 3, 5, 200, 100, 8],
                    'b' : [300, 100, 6, 8, 2, 3]})
print(77.0 * 1.5)

b3 = outliers(a3)
print(b3)




'''
# 컬럼이 1개일때

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out , [25, 75])
    print("1분사분위 : ", quartile_1)  
    print("3분사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])
# (10, )
b = outliers(a)

print("이상치의 위치 :", b)

###### 아래꺼는 틀린것 ############

c = np.array([[1,2,3,4,10000,6,7,5000,90,100],
              [1,20000,3,4,5,6,7,8,9000,100]])

# c = np.transpose()
print(c.shape)

d = outliers(c)
print(d)

print("이상치의 위치 :", b)
'''