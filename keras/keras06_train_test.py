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
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
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


# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size=1) 

print("loss : ", loss)
print("mse : ", mse)

# model.predict(x_pred)를 예측해서 y_pred로 반환한다.

y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)

# train/test
# x=수능점수, 온도, 날씨, 하이닉스, 유가 환율, 금시세, 금리 등
# y=삼성주가
# 예를들어 x의 자료가 엑셀로 365일치의 데이터로 만들어져있다면, train 7달, test 3달로 나누어서 모델을 돌려야 y값이 정확히 나올 수 있다
# model.fit에 train 값을, model.evalatae에 test 값을 넣으면 된다.
# 왜 나누는가?
# ex.수능시험 답만 외운 애들은 수능가서 망함
# 평가 데이터(검증값)는 모델에 반영이 안됨!


