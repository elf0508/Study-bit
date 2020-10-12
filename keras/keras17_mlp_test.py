# R2를 0.5 이하
#  레이어의 깊이 5개 이상 / 인풋, 아웃풋 포함
# 노드의 개수 10개 이상 / 각 레이어당
# ehoch는 30개 이상
# batch_size는 8 이하

import numpy as np 

# 1 ~ 100까지의 숫자
# x = np.array(range(1,101)) # 웨이트는 1, 바이어스는 100짜리
# x = np.array([range(1,101), range(311,411), range(100)]) # 3행 100열
# y = np.array([range(101,201),range(711,811),range(100)])

x = np.transpose([range(1,101), range(311,411), range(100)])  # 100행 3열로 바뀌었다

y = np.transpose(range(711,811))

# np = np.transpose(x)
# np = np.transpose(y)

print(x.shape)

# 데이터 분리

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state=66, shuffle=True,
#     x, y, shuffle=False,
#     test_size=0.6  
# )

x_train, x_test, y_train, y_test = train_test_split(
    # x, y, random_state=66, shuffle=True,
    x, y, shuffle=False,
    train_size=0.8 
)

# x_test, x_val, y_test, y_val = train_test_split(
#     # x_test, y_test, random_state=66, 
#     x_test, y_test,shuffle=False, 
#     test_size=0.5 
# )

# x_test, y_test = train_test_split(
#     x_test, y_test, random_state=66, 
#     x_test, y_test,shuffle=False, 
#     test_size=0.5 
# )

print(x_train)
# print(x_val)
print(x_test)


# 2. 모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim = 1))
model.add(Dense(5, input_dim = 3))  

# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(50))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(600))
model.add(Dense(600))
model.add(Dense(600))

model.add(Dense(1))


# 3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=30, batch_size=3,
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

