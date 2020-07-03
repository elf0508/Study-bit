# keras70을 카피해서 Sequential로 바꾸고

# 과적합 줄이는 방법 1.
# regularizer (레귤라이제이션) : 정규화
# kernel_regularizer

# 규제 : 가중치가 작은 값을 가지도록 가중치 값의 분포를 좀 더 균일하게 만들어 준다.
# L1 규제 : 가중치의 절대값 합
# regularizer.l1(l=0.01)
# 일부 가중치 파라미터를 0으로 만든다

# L2 규제(=weight decay) : 가중치의 제곱의 합
# regularizer.l2(l=0.01)
# 가중치 파라미터를 제한하지만 완전히 0으로 만들지는 않는다.

# loss = L1 * reduce_sum(abs(x))  절대값 전부 다 더한다.
# loss = L2 * reduce_sum(square(x))  제곱의 규제 /  레이어에 전달시 값 축소한다.

'''
L1과 L2 모두 regularization 효과로 theta(세타)를 0으로 보내고자 합니다. 
L1의 경우 theta(세타)의 크기에 상관없이 일정한 힘으로 0 방향으로 theta(세타)를 이동시킵니다. 
반면, L2의 경우 theta(세타)의 크기가 작아질수록 0으로 theta(세타)를 이동시키는 힘이 약해집니다. 
이러한 차이로 L1은 많은 feature들의 가중치를 0으로 만들고 
L2는 0으로 만들지 않고 0보다 아주 가까운 값으로 만듭니다.

일반적으로 L1은 sparse solution, 즉 대부분의 theta(세타)를 0으로 만들고 몇몇 theta(세타)를 이용해 값을 예측합니다. 
일부의 feature만으로 solution을 찾는 것입니다. 
반대로 L2는 영향력이 낮은 feature도 유지하고 이를 통해 더 좋은 예측값을 찾고자 합니다.

'''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Input
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
                 input_shape = (32, 32, 3), padding = 'same',
                 activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                 padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(256, activation = 'relu', kernel_regularizer=l2(0.001)))
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
loss :  1.0343044483184813
acc :  0.7257999777793884
'''