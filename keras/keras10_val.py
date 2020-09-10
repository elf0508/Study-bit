#validation 추가
#train만 하면 교과서만 푼 것
#train, val은 훈련하면서 검증 model.fit에서 실행이 되고 교과서 + 모의고사를 같이 푼 것
#다시말해, 훈련시키고 컨닝(1epoch), 훈련시키고 컨닝(2epoch), ... 반복적인 행위가 됨
#test는 model.evaluate에서 실행


# 1. 데이터

import numpy as np 

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

# x_pred = np.array([16,17,18])

x_val = np.array([101,102,103,104,105])  # val(validation) : 확인
y_val = np.array([101,102,103,104,105])


# 2. 모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim = 1))

model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(300))

model.add(Dense(1))


# 3. 훈련

# train 데이터 > test 데이터

#val 값은 train과 같이 훈련되어야 하기에 model.fit에 포함

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=300, batch_size=1,
            validation_data=(x_val, y_val))


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

