# 1. 데이터
import numpy as np 

# 1 ~ 100까지의 숫자
x = np.array(range(1,101))
y = np.array(range(101,201))
# 데이터 분리
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state=66, shuffle=True,
#     x, y, shuffle=False,
#     test_size=0.6  
# )
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True,
    # x, y, shuffle=False,
    train_size=0.6  
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

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]

# y_train = x[:60]
# y_val = x[60:80]
# y_test = x[80:]

print(x_train)
# print(x_val)
print(x_test)


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1,
           validation_split=0.3)

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


