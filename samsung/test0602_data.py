import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/samsung.csv',
                        index_col=0, # None # '일자'가 데이터로 잡힌다.
                        header=0, 
                        sep=',',
                        encoding='CP949')

hite = pd.read_csv('./data/csv/hite.csv',
                        index_col=0, # None
                        header=0,
                        sep=',',
                        encoding='CP949')

print(samsung)
print(samsung.head())
print(hite.head())  # 결측치에 값 넣어야 된다.
print(samsung.shape)  # (700, 1)
print(hite.shape)  # (720, 5)

# None(결측치) 제거 1
samsung = samsung.dropna(axis=0)  # 0 을 넣으면, 행 삭제
print(samsung)  # [509 rows x 1 columns]
print(samsung.shape)  # ( , 1)
hite = hite.fillna(method='bfill')  # bfill = 전날 값으로 채우겠다 
hite = hite.dropna(axis=0)

# None(결측치) 제거 2
# hite = hite[0:509]  
# hite.iloc[0, 1:5] = [10,20,30,40] # iloc = 숫자를 넣는다 / 끝에 값에서 빼기 1 
# hite.loc['2020-06-02', '고가':'거래량'] = ['10','20','30','40'] # loc = "행", "열"

print(hite)  # [509 rows x 5 columns]

# 정렬 바꾸기- 오름차순(ascending)으로 변경하기

samsung = samsung.sort_values(['일자'], ascending=['True'])
hite = hite.sort_values(['일자'], ascending=['True'])

print(samsung)
print(hite)

# 콤마제거, 문자를 상수로 형변환

for i in range(len(samsung.index)):  # '37,000' -> 37000
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))

print(samsung)
print(type(samsung.iloc[0,0])) # <class 'int'>

for i in range(len(hite.index)):  # i : 전체 행 / j : 전체 열
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',""))

print(hite)
print(type(hite.iloc[1,1])) # <class 'int'>

print(samsung.shape)  # (509, 1)  509바이 1 (590행 1열)
print(hite.shape)   # (509, 5)  509바이 5 (590행 5열)

samsung = samsung.values
hite = hite.values

print(type(samsung))  # <class 'numpy.ndarray'>
print(type(hite))  # <class 'numpy.ndarray'>

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)

# numpy파일 까지 만들어서, 저장했다.


