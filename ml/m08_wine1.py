# 머신러닝으로 만들기
# 다중분류

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils

standardscaler1 = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()
pca = PCA(n_components = 10)

### 1. 데이터
wine = pd.read_csv('./data/csv/winequality-white.csv',
                   header = 0, index_col = None,
                   sep = ';', encoding = 'cp949')
print(wine.head())
print(wine.tail())
print(wine.shape)                   # (4898, 12)

## 1-1. 데이터 전처리
# 1-1-1. 결측치 확인
print("결측치 확인 : ", wine.isna())                  # 확인 ok


## 1-2. numpy 파일로 변환 후 저장
wine = wine.values
print(type(wine))                   # <class 'numpy.ndarray'>
print("wine : ", wine)
print("wine.shape : ", wine.shape)                   # (4898, 12)
np.save('./data/wine_np.npy', arr = wine)

## 1-3. numpy 파일 불러오기
np.load('./data/wine_np.npy')
print(wine.shape)                   # (4898, 12)


## 1-4. 데이터 나누기
x = wine[:, 0:11]                   #0~11행
y = wine[:, -1:]                    #마지막 1개의 행
print(x.shape)                      # (4898, 11)
print(y.shape)                      # (4898, 1)

'''
# 3) x, y 나누기
print(type(wine))
x = wine[ : , :-1] # -1 : 뒤에서부터 자르겠다/ :-1은 뒤에서부터 앞까지 전체 다.
y = wine[ : ,  -1] # -1 : 뒤에꺼 1개만 하겠다.

'''

## 1-5. train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.25)
print(x_train.shape)                # (4408, 11)
print(x_test.shape)                 # (490, 11)
print(y_train.shape)                # (4408, 1)
print(y_test.shape)                 # (490, 1)

## 1-6. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (4408, 10)
print(y_test.shape)                 # (490, 10)


## 1-7. 데이터 Scaling  Standard Scaler
standardscaler1.fit(x_train)
x_train = standardscaler1.transform(x_train)
x_test = standardscaler1.transform(x_test)
print(x_train[0])                   # [0.33653846 0.21621622 0.25903614 0.01687117 0.24315068 0.12543554
                                    #  0.31888112 0.06499518 0.41666667 0.23255814 0.77419355]
print(x_test[1])                    # [0.40384615 0.10810811 0.29518072 0.01840491 0.17808219 0.04878049
                                    #  0.38041958 0.13635487 0.4537037  0.30232558 0.32258065]

## 1-8. PCA
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)                # (4408, 10)
print(x_test.shape)                 # (490, 10)


# 2. 모델링
model = RandomForestClassifier(n_estimators = 25)
# model = SVC()
# model = KNeighborsClassifier()


# 3. 모델 훈련
model.fit(x_train, y_train)


# 4. 모델 평가
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("분류 정확도 : ", acc)                # 0.5420408163265306

'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

wine = pd.read_csv('winequality-white.csv',sep=';')

x = wine.iloc[:,0:-1]
y = wine.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(
    x, y, random_state=66, test_size=0.2 )

model = RandomForestClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test) # 회귀던 분류던 사용할 수 있음
print('RandomForestClassifier score(acc)는',score)

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_predict,y_test)
# print('RandomForestClassifier acc는',acc)

'''