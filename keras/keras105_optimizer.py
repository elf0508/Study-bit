# 1.데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

# 2.모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10, input_dim = 1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))

model.add(Dense(1))

# model.summary()

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

 # lr = learning_rate

optimizer = Adam(lr=0.001)     # loss :  [0.03224189579486847, 0.03224189579486847] / [[3.4250028]]
# optimizer = RMSprop(lr=0.001)  # loss :  [0.0118400938808918, 0.0118400938808918] / [[3.448438]]
# optimizer = SGD(lr=0.001)      # loss :  [0.051301490515470505, 0.051301490515470505] / [[3.3823326]]
# optimizer = Adadelta(lr=0.001)   # loss :  [11.759016036987305, 11.759016036987305] / [[-0.8826536]]
# optimizer = Adagrad(lr=0.001)    # loss :  [7.641761779785156, 7.641761779785156] / [[-0.05400188]]
# optimizer = Nadam(lr=0.001)      # loss :  [1.2523717880249023, 1.2523717880249023] / [[2.0051987]]

# optimizer 설명 -->  https://gomguard.tistory.com/187

model.compile(loss='mse', optimizer = optimizer, metrics=['mse'])

model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)

print("loss : ", loss)

pred1 = model.predict([3.5])

print(pred1)
