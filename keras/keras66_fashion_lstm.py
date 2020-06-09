# 과제 4

# Sequential형으로
# 하단에 주석으로 acc, loss 결과 명시

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_test.shape)   # (10000,)

# plt.imshow(x_train[0])
# plt.show()

print(x_train[0].shape)     # (28, 28)                          


# 데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)           # (60000, 10)

# 데티어 전처리 2. 정규화                                            

x_train = x_train.reshape(60000, 28*28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28*28, 1).astype('float32') /255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 모델 구성

model = Sequential()  
model.add(LSTM(20,activation='relu', input_shape = (28*28, 1)))                            
model.add(Dense(35,activation='relu' ))                                             
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(650, activation='relu'))                                              
model.add(Dense(850, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(200, activation='relu')) 
model.add(Dropout(0.2))                                                    
model.add(Dense(10))                                                       

model.summary()


# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, 
         epochs=20, batch_size=256,
        validation_split=0.25, verbose=1)  
        # 콜백에는 리스트 형태 



# 4. 평가

loss, acc = model.evaluate(x_test,  y_test)

print('loss :', loss )
print('acc :', acc )

# predict = model.predict(x_test)
# print(predict)
# print(np.argmax(predict, axis = 1))


# predict = model.predict(x_test)   # 'softmax'가 적용되지 않은 모습으로 나옴

# y_pred = model.predict(x_test)   # 'softmax'가 적용되지 않은 모습으로 나옴
# print(y_pred)
# print(np.argmax(y_pred, axis = 1))

# print('y_pred : ', y_pred)  
# print(y_pred.shape)                              
