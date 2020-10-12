# import할 케라스 api는 상단에 기술한다.

from keras.models import Sequential
from keras.layers import Dense
import numpy as np 


# 데이터 준비

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])


# 모델 구성

model = Sequential()

model.add(Dense(5, input_dim = 1, activation='relu'))
# input은 1개이고, output은 5이다.

model.add(Dense(3))
# input은 5개이고, output은 3이다.

model.add(Dense(2))
model.add(Dense(1, activation='relu'))
# Denses는 keras의 DNN 기본구조이다.

model.summary()
# 까지만 했을 때의 결과값에서
# Output Shape는 레이어에 있는 노드의 갯수를 나타내고, Param은 신경망의 라인 갯수를 나타낸다.
# Param의 결과값은 (노드의 수) * (input의 차원) + (노드의 수)를 나타낸다.

'''
# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=900, batch_size=1, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

# 평가, 예측
print("loss :", loss)
print("acc :", acc)

output = model.predict(x_test)
print("결과물 : \n", output)
'''



