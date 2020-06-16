# 10가지 이미지를 찾아서, 칼라(3)

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
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

print(x_train[0].shape)     # (32, 32, 3)                          


# 데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)           # (50000, 10)

# 데티어 전처리 2. 정규화                                            
# x_train = x_train.reshape(60000, 28, 28).astype('float32') /255  
# x_test = x_test.reshape(10000, 28, 28).astype('float32') /255   # 뒤에 ' . '을 써도 된다.                                  
#             cnn 사용을 위한 4차원       # 타입 변환       # (x - min) / (max - min) : max =255, min = 0                                      
#                                         : 나누면 소수점이 되기때문에 int형 -> float형으로 타입변환

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') /255  
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') /255

print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  # (10000, 32, 32, 3)
print(y_train.shape) # (50000, 10)
print(y_test.shape)  # (10000, 10)

# 모델 구성

input1 = Input(shape=(32,32,3))
coni_1 = Conv2D(450, (2, 2))(input1)                            
coni_2 = Conv2D(650, (3, 3))(coni_1)   
drop = Dropout(0.2)(coni_2)

coni_3 = Conv2D(750, (2, 2), padding = 'same')(drop) 
                                            
flatten = Flatten()(coni_3)                                                   
coni_4 = Dense(10, activation='softmax')(flatten)                            

model = Model(inputs=input1, outputs = coni_4)

model.summary()

# loss : 1.8287592342376708
# acc : 0.37220001220703125

# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, 
         epochs=10, batch_size=256,
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
       
