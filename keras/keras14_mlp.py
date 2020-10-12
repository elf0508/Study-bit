# 데이터 두개 이상을 사용해보자
# ex) data = 삼성, 하이닉스 주가 output = 다우지수, xx지수 (output은 2개 이상 나올 수 있음)
# _행 _열  --> _행 _열로 바꿔야한다. 예) 3행 100열 --> 100행 3열

# 1. 데이터

import numpy as np 

# 1 ~ 100까지의 숫자
# x = np.array(range(1,101)) # 웨이트는 1, 바이어스는 100짜리
x = np.array([range(1,101), range(311,411), range(100)]) 
y = np.array([range(101,201),range(711,811),range(100)])

# 파이썬에는 list가 있는데 덩어리를 []로 묶어줘야함
# 1st 덩어리 : w=1, b=100
# 2nd 덩어리 : w=1, b=400
# 3rd 덩어리 : w=1

print(x.shape)
print(y.shape)  
 # (3 , 100)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape) 
print(y.shape)
# (100, 3)
# 열우선, 행무시


# 데이터 분리

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    # x, y, random_state=66, shuffle=True,
    x, y, shuffle=False,
    train_size=0.8
     # (80, 3)
)

print(x_train)
# print(x_val)
print(x_test)


# 2. 모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# 여지껏 input_dim=1이었지만, 데이터 컬럼이 3개 이므로, input_dim=3으로 변경
model.add(Dense(5, input_dim = 3)) 

model.add(Dense(3))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(500))

model.add(Dense(3))  
# 역시나 output_Dense도 3으로 변경 


# 3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=30, batch_size=1,
           validation_split=0.25)  
            # x_train : (60, 3) ,  x_val : (20, 3), x_test : (20, 3)
            
# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size=1) 

print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
# print(y_predict)


# RMSE 구하기

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):

     return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))


# R2 구하기

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)

print("R2 : ", r2)


