# 과제 : R2를 음수가 아닌 0.5 이하로 줄이기
# 레이어는 인풋과 아웃풋을 포함 5개 이상(히든이 3개 이상), 히든 레이어에 있는 노드는 레이어당 각각 5개 이상
# batch_size = 1
# epochs = 100 이상

# 1. 데이터

import numpy as np 

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = np.array([16,17,18])


# 2. 모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim = 1))

model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(35))
model.add(Dense(75))
model.add(Dense(75))
model.add(Dense(75))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(500))

model.add(Dense(1))


# 3. 훈련

# train 데이터 > test 데이터

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1)


# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size=1) 

print("loss : ", loss)
print("mse : ", mse)

# model.predict(x_pred)를 예측해서 y_pred로 반환한다.
# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)

y_predict = model.predict(x_test)

print(y_predict)


# RMSE 구하기

# 재사용을 위한 함수 만들기 : def라고 선언한다. 함수명 : RMSE

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):

     return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))


# R2 구하기

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)

print("R2 : ", r2)