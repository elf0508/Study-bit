# normalization (노말라이제이션)
# 0 ~ 1 사이로 수렴 시키겠다.

'''
배치 정규화(BN, Batch Normalization)
: 각 층의 활성화 함수의 출력값 분포가 골고루 분포되도록 '강제'하는 방법으로, 
각 층에서의 활성화 함수 출력값이 정규분포(normal distribution)를 이루도록 하는 방법이다. 

즉, 학습하는 동안 이전 레이어에서의 가중치 매개변수가 변함에 따라 
활성화 함수 출력값의 분포가 변화하는 내부 공변량 변화(Internal Covariate Shift) 문제를 
줄이는 방법

미니배치(mini-batch)를 단위로 데이터의 분포가 평균(​, mean)이 0, 
분산(​, variance)이 1이 되도록 정규화(normalization)한다.

관련 설명 : https://excelsior-cjh.tistory.com/178

관련 설명 : https://jsideas.net/batch_normalization/

관련 유튜브 : https://www.youtube.com/watch?v=DtEq44FTPM4

텐서플로에서 Batch Normalization 구현하기

import 방법 --> tf.layers.batch_normalization

'''

# 과적합 피하기 3.
# Dropout

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Input, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils
# from keras.datasets import cifar100
from keras.datasets import cifar10
from keras.optimizers import Adam
# from keras.regularizers import l1, l2, l1_l2


modelfath = './model/cifar10_{epoch:02d} - {val_loss:.4f}.hdf5'

# 클래스 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
cp = ModelCheckpoint(filepath = modelfath, monitor = 'val_loss',
                     mode = 'auto', save_best_only = True)
# tb_hist = TensorBoard(log_dir = './graph', histogram_freq = 0,
#                       write_graph = True, write_images = True)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)            # (50000, 32, 32, 3)
print(x_test.shape)             # (10000, 32, 32, 3)
print(y_train.shape)            # (50000, 1)
print(y_test.shape)             # (10000, 1)

# 1-1. 정규화
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0

# 1-2. OHE
# y_train = np_utils.to_categorical(y_train, num_classes = 100)
# y_test = np_utils.to_categorical(y_test, num_classes = 100)
# print(y_train.shape)
# print(y_test.shape)


# 2. 모델링

from keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 input_shape = (32, 32, 3), padding = 'same'))
                 
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                 padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'sparse_categorical_crossentropy',         # 원핫인코딩을 하지 않았을 때, 다중분류 손실함수
              metrics = ['accuracy'],                           # sparse는 개인 취향이다!
              optimizer = Adam(1e-4))                           # 0.0001
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 32,
                 validation_split = 0.3, verbose = 1)

print(hist.history.keys())

# 4. 모델 평가
res = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", res[0])
print("acc : ", res[1])



# 5. 시각화
plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.title('loss')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['accuracy'], marker = '.', c = 'violet', label = 'acc')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'green', label = 'val_acc')
plt.title('accuracy')
plt.grid()
plt.ylim(0, 1.0)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'lower right')
plt.show()


'''
loss :  1.5908753252983092
acc :  0.652899980545044

'''