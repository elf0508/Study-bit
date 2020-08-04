# x를 4차원에서 2차원으로 변형, Dense 모델에 넣어주기
# keras 56_mnist_DNN.py 복붙

import numpy as np


#1. 데이터
from tensorflow.keras.datasets import mnist

mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                              # (60000, 28, 28)
print(x_test.shape)                               # (10000, )
print(y_train.shape)                              # (60000, )
print(y_test.shape)                               # (10000, )


# x_data전처리 : MinMaxScaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# # y_data 전처리 : one_hot_encoding (다중 분류)
# from keras.utils.np_utils import to_categorical
# y_trian = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape)


# reshape : Dense형 모델 사용을 위한 '2차원'
x_train = x_train.reshape(60000, 28*28 ) 
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape)                              # (60000, 784)
print(x_test.shape)                               # (10000, 784)



#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

input_img = Input(shape=(784, ))
# encoded = Dense(16, activation = 'relu')(input_img)
encoded = Dense(32, activation = 'relu')(input_img)
# encoded = Dense(64, activation = 'relu')(input_img)
decoded = Dense(784, activation = 'sigmoid')(encoded)   # 전처리 해주었기 때문에 0 ~ 1사이의 값 가짐

autoencoder = Model(input_img, decoded)

autoencoder.summary()


#3. 훈련
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
# autoencoder.compile(optimizer = 'adam', loss = 'mse')

autoencoder.fit(x_train, x_train, epochs = 50, batch_size = 256,                                  # 앞, 뒤가 똑같은
                            validation_split =0.2)

dencoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(dencoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



'''
# x를 4차원에서 2차원으로 변형, Dense 모델에 넣어주기
# keras 56_mnist_DNN.py 복붙


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

#Datasets 불러오기
from tensorflow.keras.datasets import mnist  

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])


print('y_train : ' , y_train[0])


print(x_train.shape)                   #(60000, 28, 28)
print(x_test.shape)                    #(10000, 28, 28)
print(y_train.shape)                   #(60000,) 스칼라, 1 dim(vector)
print(y_test.shape)                    #(10000,)


print(x_train[0].shape) #(28,28) 짜리 
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()



# 1. y-------------------------------------------------------------------
# Data 전처리 / 1. OneHotEncoding 큰 값만 불러온다 y
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10) 6만장 10개로 증폭/ 아웃풋 dimension 10

# 0과 255 사이를 0과 1 사이로 바꿔줘

x_train = x_train / 255


# 2. x-------------------------------------------------------------------

# Data 전처리/ 2. 정규화 x
# 형을 실수형으로 변환
# # MinMax scaler (x - 최대)/ (최대 - 최소)

############ 4차원을 2차원으로#########
x_train = x_train.reshape(60000, 784).astype('float32')/255 ##??????
x_test  = x_test.reshape (10000, 784).astype('float32')/255 ##??????
######################################
print('x_train.shape: ', x_train.shape)
print('x_test.shape : ' , x_test.shape)


#2. 모델구성 ==========================================

# 함수형

input_img = Input(shape= (784, ))
encoded = Dense(32, activation='relu')(input_img) # 784개 중 특성 32개 추출
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()
autoencoder.compile(optimizer='adam', loss= 'binary_crossentropy')
autoencoder.compile(optimizer='adam', loss= 'mse')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2) # y값이 x값이 됨

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


model = Sequential()

model.add(Dense(100, activation='relu', input_shape =(784, )))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) ## softmax 꼭 써야해!!

model.summary()

# 3. 실행 ===================================

model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 150, batch_size= 256, verbose=1,  validation_split = 0.2)

#4. 평가, 예측 ========================================
loss, acc = model.evaluate(x_test, y_test, batch_size=256)

print('loss: ', loss)
print('acc:', acc)

############################

# 20-05-28_14 0900~
# keras54 pull
# mnist DNN 으로 모델 구성
# acc = 0.98 이상 결과 도출

#  튜닝 값 (0.98 이상)
#  <epoch:225, batch:512>
#  loss : 0.15608389074802398
#  acc : 0.9801999926567078
  

import numpy as np

# Datasets 불러오기
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])                   # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train : ', y_train[0])     # 5

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)


# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(50, activation='relu', input_dim = 784))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=50, mode = 'auto')

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=2, validation_split=0.2,
                 callbacks=[es])

# matplotlib 사용
 # import matplotlib.pyplot as plt

 # plt.plot(hist.history['loss'])
 # plt.plot(hist.history['val_loss'])
 # plt.plot(hist.history['acc'])
 # plt.plot(hist.history['val_acc'])
 # plt.title('loss & acc')
 # plt.ylabel('loss, acc')
 # plt.xlabel('epoch')
 # plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
 # plt.show()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=256)
print("loss :", loss)
print("acc :", acc)

'''