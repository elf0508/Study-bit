# CNN - 다중분류

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 

mnist.load_data()                                         # mnist파일 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist에서 이미 x_train, y_train으로 나눠져 있는 값 가져오기

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)        : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                                       # (10000,)


print(x_train[0].shape)                                   # (28, 28)
# plt.imshow(x_train[0], 'gray')                          # '2차원'을 집어넣어주면 수치화된 것을 이미지로 볼 수 있도록 해줌    
# plt.imshow(x_train[0])                                  # 색깔로 나옴
# plt.show()                                              # 0 그림으로 보여주기


# 데이터 전처리 1. 원핫인코딩

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화                                            
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.   # 뒤에 ' . '을 써도 된다.                                  
#             cnn 사용을 위한 4차원       # 타입 변환       # (x - min) / (max - min) : max =255, min = 0                                      
#                                         : 나누면 소수점이 되기때문에 int형 -> float형으로 타입변환
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# 모델 구성

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D  # Convolution 2D(가로 * 세로) : 이미지 자르는 것 
from keras.layers import Dense, Flatten

model = Sequential() # 이미지를 가로, 세로 2, 2로 자르겠다. 

model.add(Conv2D(10, (2, 2), input_shape = (28, 28, 1)))                  # (N, 9, 9, 10)   

model.add(Conv2D(7, (3, 3)))                                              # (N, 7, 7, 7)
model.add(Conv2D(5, (2, 2), padding = 'same'))                            # (N, 7, 7, 5)
model.add(Conv2D(5, (2, 2)))                                              # (N, 6, 6, 5)
# model.add(Conv2D(5, (2, 2), strides = 2))                               # (N, 3, 3, 5)
# model.add(Conv2D(5, (2, 2), strides = 2, padding ='same'))              # (N, 3, 3, 5) : padding보다 strides가 우선
# model.add(MaxPooling2D(pool_size = 2))                                  # (N, 3, 3, 5)
model.add(Flatten())                                                      # (N, 45)      : 2차원

model.add(Dense(10, activation='softmax'))                                # (N, 1)

model.summary()

# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, 
         epochs=30, batch_size=1,
        validation_split=0.25, verbose=1)  
        # 콜백에는 리스트 형태 


# 4. 평가

loss, acc = model.evaluate(x_test,  y_test)

print('loss :', loss )
print('acc :', acc )



# predict = model.predict(x_test)
# print(predict)
# print(np.argmax(predict, axis = 1))


# 4. 예측

# predict = model.predict(x_test)   # 'softmax'가 적용되지 않은 모습으로 나옴

# y_pred = model.predict(x_test)   # 'softmax'가 적용되지 않은 모습으로 나옴

# print('y_pred : ', y_pred)  

# print(y_pred.shape)                              
