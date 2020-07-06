# reuters : 뉴스 기사로 카테고리를 나누는 것 / 46개의 카테고리
# 1만개 --> 8천개로 훈련 --> 카테고리별로 나누는 것

from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.데이터
# (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 1000, test_split = 0.2)
# (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 10000, test_split = 0.2)
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 15000, test_split = 0.2)

print(x_train.shape, x_test.shape)  # (8982, ) (2246, )
print(y_train.shape, y_test.shape)  # 8982, ) (2246, )

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))   # 87   크기가 일정하지 않다. --> 빈 자리는 0으로 채운다.

# y의 카테고리 개수 출력
category = (np.max(y_train)) + 1
print("카테고리 : ", category)   #  카테고리 :  46  / 0 ~ 45까지 있다. /다중분류

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# 판다스 / groupby
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)

# 주간 과제 : groupby()의 사용법 숙지할것

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 100, padding = 'pre') 
x_test = pad_sequences(x_test, maxlen = 100, padding = 'pre') 
# padding = 'pre' --> 앞에서부터 0으로 13개 채운다.

# print(len(x_train[0]))
# print(len(x_train[-1]))

# 원핫인코딩

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)   # (8982, 100) (2246, 100)

# 2.모델

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()

# model.add(Embedding(1000, 128, input_length = 100))
# model.add(Embedding(10000, 128))
model.add(Embedding(15000, 128))

model.add(LSTM(64))
# model.add(LSTM(100))

model.add(Dense(46, activation= 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                    metrics = ['acc'])

history = model.fit(x_train, y_train, batch_size = 100, epochs = 30,
                        validation_split = 0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)

# 그래프
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker ='.', c ='red', label ='TestSet Loss')
plt.plot(y_loss, marker ='.', c ='blue', label ='TrainSet Loss')
plt.legend(loc ='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()

# acc :  0.6237756013870239  1천
# acc :  0.6438112258911133  1만
# acc :  0.6544969081878662  1만 5천



