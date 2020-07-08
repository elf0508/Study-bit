# 과제 2

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

# 데이터 전처리 2. 정규화                                            

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 모델 구성 Sequential이면서 Conv2D 추가

model = Sequential()  
model.add(Conv2D(120, (2, 2), input_shape = (28, 28, 1)))                            
model.add(Conv2D(170, (2, 2)))                                              
model.add(Conv2D(350, (2, 2), padding = 'same'))                            
model.add(Conv2D(500, (2, 2))) 
model.add(Dropout(0.2))

model.add(Conv2D(70, (2, 2))) 
model.add(Conv2D(90, (2, 2))) 
# model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size = 2))                                    
model.add(Flatten())                                                     
model.add(Dense(10))                                                       

model.summary()

# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=1, mode='auto')

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

# loss : 6.598748334503174
# acc : 0.0034000000450760126