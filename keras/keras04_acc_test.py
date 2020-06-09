# 1. 데이터
import numpy as np 
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])

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
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# mse (회귀지표 - 수치로 나온다/ 평균 제곱 오차): 오차의 제곱에 대해 평균을 취한 것이며, 
# 작을 수록 원본과의 오차가 적은 것이므로 추측한 값의 정확성이 높은 것이다. 
# 통계적 추정의 정확성에 대한 질적인 척도로 많이 사용된다.
# 표본정보를 근거로 하여 알지 못하는 모수의 참값을 하나의 수치로 추론하는 것
# acc (분류지표) : y값이 고정이 되어있어야 함 / 결과값이 정해져 있어야 함
# loss='mse' 로 해서 y값이 고정값이 아닌, 자유로운 값이 나온것이다
# metrics 는 평가한 것을 보여주는 값 (반환해주는 값)

model.fit(x, y, epochs=30, batch_size=1)

# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1) 
print("loss : ", loss)
print("acc : ", acc)

# model.predict(x_pred)를 예측해서 y_pred로 반환한다.
y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)

# #why? 회귀와 분류방식
# 1. 회귀 : 수치를 넣었을 때 수치로 답을 한다
# 2. 분류 : 강아지/고양이 _0과 1_ y값이 정해져있어야 함 ex) 동전을 던지고 난 결과
#loss=mse, metrics=acc 이기 때문에 잘못된 모델을 쓴것임