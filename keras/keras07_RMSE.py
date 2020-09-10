# RMSE : 평균 제곱근 오차 / 입력변수가 여러 개일때, 오차가 가장 작은 선(기울기)을 찾는 것 = 값이 작을수록 정밀하다.

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
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))

model.add(Dense(1))


# 3. 훈련

# train 데이터 > test 데이터

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=30, batch_size=1)
# fit : 최적의 w 값을 찾아라


# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size=1) 

print("loss : ", loss)
print("mse : ", mse)

# model.predict(x_pred)를 예측해서 y_pred로 반환한다.
# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)

y_predict = model.predict(x_test)  # RMSE의 계산을 위한  x_test의 예측값 


# 입력값에 y_test는 안되나요? ★절대안됨! x = input, y = output 이기 때문에!
print(y_predict)


# 재사용을 위한 함수 만들기 : def라고 선언한다. 사용자의 개입없이 자동으로 할당되는 설정 또는 값 
# 함수명 : RMSE
from sklearn.metrics import mean_squared_error

# 사이킷런 = 케라스, 텐서플로우 이전의 킹왕짱
#사이킷런에는 RMSE가 없어서 mse를 불러옴 -> 루트 씌울거임

def RMSE(y_test, y_predict): 
     return np.sqrt(mean_squared_error(y_test, y_predict))
     
print("RMSE : ", RMSE(y_test, y_predict))

# def RMSE(y_test, y_predict): <-- def = 함수를 호출하겠다 () = 입력값
#      return np.sqrt(mean_squared_error(y_test, y_predict)) <-- 반환 = 출력값
                 #sqrt = 루트
            #mse=(실제값-예측값)^2 / n 한 후 루트이기에 y_test, y_predict가 입력값이 됨
# print("RMSE : ", RMSE(y_test, y_predict)) 
# ("  ", : 콘솔창에 print 해서 보여줘라, 뒤에 함수들 (    )))을 실행해서)
# R2 = 1에 가까울수록 좋은 값. accuracy와 유사한 결과를 도출
# mse, RMSE는 값이 작을수록 좋은 값




