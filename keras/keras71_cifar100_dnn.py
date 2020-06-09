# 100가지 이미지를 찾아서, 칼라(3)
# DNN

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model, Input
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = cifar100.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)  # (50000, 32, 32, 3)
print(x_test.shape)   # (10000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
print(y_test.shape)   # (10000, 1)

plt.imshow(x_train[0])
# plt.show()

# 데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)           # (50000, 100)

# 데이터 전처리 2. 정규화                                            

x_train = x_train.reshape(50000, 32*32*3 ).astype('float32') /255  
x_test = x_test.reshape(10000, 32*32*3 ).astype('float32') /255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 모델 구성 # Sequential 이면서 Dense형

model = Sequential() 

model.add(Dense(10,input_shape=(32*32*3,)))
model.add(Dropout(0.2))

model.add(Dense(30,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(210,activation='relu'))
model.add(Dense(310,activation='relu'))
model.add(Dense(410,activation='relu'))
model.add(Dropout(0.6))

model.add(Dense(210,activation='relu'))
model.add(Dense(310,activation='relu'))
model.add(Dense(110,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size = 2))                                          
flatten = Flatten()                                                   
model.add(Dense(100, activation='softmax'))                            

model.summary()

# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

from keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint =  ModelCheckpoint(filepath = modelpath, monitor='val_loss',
                            save_best_only=True, mode='auto')

es = EarlyStopping(monitor = 'loss', patience = 2, mode= 'auto')

""" Tensorboard """
from keras.callbacks import TensorBoard   # Tensorboard 가져오기
tb_hist = TensorBoard(log_dir='graph', histogram_freq= 0 , # log_dir=' 폴더 ' : 제일 많이 틀림
                      write_graph= True, write_images= True) 

hist = model.fit(x_train, y_train, 
                epochs=10, batch_size=256,
                validation_split=0.25, verbose=1,
                callbacks = [es, checkpoint, tb_hist])  
        # 콜백에는 리스트 형태 



# 4. 평가

loss_acc = model.evaluate(x_test,  y_test, batch_size= 64)

loss = hist.history['loss']   # model.fit 에서 나온 값
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc: ', acc)                               
print('val_acc: ', val_acc)
print('loss_acc: ', loss_acc)                     
                    

import matplotlib.pyplot as plt    

plt.figure(figsize = (10, 6))  # 10 x 6인치의 판이 생김

# 1번 그림
plt.subplot(2, 1, 1)  # (2, 1, 1) 2행 1열의 그림 1번째꺼 / subplot : 2장 그림               
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')                     
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')                  
plt.grid()     # 격자 생성
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss','val_loss']) 
plt.legend(loc='upper right')   

# 2번 그림
plt.subplot(2, 1, 2)   # (2, 1, 2) 2행 1열의 그림 2번째꺼               
plt.plot(hist.history['acc'])                     
plt.plot(hist.history['val_acc'])                  
plt.grid()       # 격자 생성
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()  

# acc:  [0.00992, 0.010773334, 0.015546666, 0.01824, 0.01824, 0.019813333, 0.0216, 0.022426667, 0.023066666, 0.023866666]
# val_acc:  [0.009279999881982803, 0.010239999741315842, 0.01640000008046627, 0.01648000068962574, 0.020239999517798424, 0.019999999552965164, 0.02199999988079071, 0.02151999995112419, 0.023679999634623528, 0.024960000067949295]
# loss_acc:  [4.421079733276367, 0.025200000032782555]

