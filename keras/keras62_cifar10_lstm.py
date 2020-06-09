# 10가지 이미지를 찾아서, 칼라(3)

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)  # (50000, 32, 32, 3)
print(x_test.shape)   # (10000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
print(y_test.shape)   # (10000, 1)

# plt.imshow(x_train[0])
# plt.show()

print(x_train[0].shape)  # (32, 32, 3)

# 데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)          # (50000, 10)

# 데티어 전처리 2. 정규화                                            
x_train = x_train.reshape(50000, 32*32, 3).astype('float32') /255  
x_test = x_test.reshape(10000, 32*32, 3).astype('float32') /255                                     

print(x_train.shape)
print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 모델 구성

model = Sequential() 
model.add(LSTM(30, activation='relu', input_shape = (32*32, 3)))                        
model.add(Dense(50))

model.add(Dropout(0.2))
                                                  
model.add(Dense(10, activation='softmax'))                                

model.summary()

# loss : nan
# acc : 0.10000000149011612

# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, 
         epochs=30, batch_size=256,
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
# print(y_pred)
# print(np.argmax(y_pred, axis = 1))

# print('y_pred : ', y_pred)  
# print(y_pred.shape)                              


