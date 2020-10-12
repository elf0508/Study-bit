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
#model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))

model.add(Dense(1))


# 3. 훈련

# train 데이터 > test 데이터

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=30, batch_size=1)  # w 가중치가 나옴


# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size=1) 

print("loss : ", loss)
print("mse : ", mse)

# model.fit에 나온 가중치로 mse를 산출 
# 그래서 y_test_predict(예측값) 나옴

# model.predict(x_pred)를 예측해서 y_pred로 반환한다.
# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)

y_predict = model.predict(x_test)  # 입력값에 y_test는 안되나요? ★절대안됨! x = input, y = output 이기 때문에!

print(y_predict)


# RMSE 구하기

# 재사용을 위한 함수 만들기 : def라고 선언한다. 함수명 : RMSE

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))


# R2 구하기

from sklearn.metrics import r2_score 

r2 = r2_score(y_test, y_predict)  #RMSE와 같이 비교하는 것이기에 r2_score에 입력값 같게

print("R2 : ", r2)


