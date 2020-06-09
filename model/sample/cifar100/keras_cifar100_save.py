# 100가지 이미지를 찾아서, 칼라(3)
# CNN

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
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

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') /255  
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') /255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 모델 구성 함수형

input1 = Input(shape=(32,32,3))
coni_1 = Conv2D(10, (2, 2))(input1)                            
coni_2 = Conv2D(15, (3, 3))(coni_1)   
drop1 = Dropout(0.2)(coni_2)

coni_3 = Conv2D(30, (3, 3))(drop1)   
coni_4 = Conv2D(40, (3, 3))(coni_3)   
coni_5 = Conv2D(50, (3, 3))(coni_4)
drop2 = Dropout(0.5)(coni_5)

coni_6 = Conv2D(210, (3, 3))(drop2) 
coni_7 = Conv2D(310, (3, 3))(coni_6)   
coni_8 = Conv2D(410, (3, 3))(coni_7)   
drop3 = Dropout(0.7)(coni_8)

coni_9 = Conv2D(210, (3, 3))(drop3)
drop4 = Dropout(0.2)(coni_9)

coni_10 = Conv2D(310, (3, 3))(drop4)   
coni_11 = Conv2D(110, (3, 3))(coni_10)
drop5 = Dropout(0.3)(coni_11)

coni_12 = Conv2D(30, (3, 3))(drop5)   
coni_13 = Conv2D(20, (3, 3),padding = 'same')(coni_12)   
drop6 = Dropout(0.5)(coni_13)

coni_14 = Conv2D(10, (2, 2), padding = 'same')(drop6) 
drop7 = Dropout(0.3)(coni_14)

                                            
flatten = Flatten()(drop7)                                                   
coni_15 = Dense(100, activation='softmax')(flatten)                            

model = Model(inputs=input1, outputs = coni_15)

model.summary()

# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

from keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint =  ModelCheckpoint(filepath = modelpath, monitor='val_loss',
                            save_best_only=True, mode='auto')

es = EarlyStopping(monitor = 'loss', patience = 20, mode= 'auto')

""" Tensorboard """
from keras.callbacks import TensorBoard   # Tensorboard 가져오기
tb_hist = TensorBoard(log_dir='graph', histogram_freq= 0 , # log_dir=' 폴더 ' : 제일 많이 틀림
                      write_graph= True, write_images= True) 


from keras.callbacks import ModelCheckpoint

modelpath = './model/sample/cifar100/cifar100_checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
                            verbose =1,
                            save_best_only= True, save_weights_only= False)   # 훈련 위에

hist = model.fit(x_train, y_train, 
                epochs=10, batch_size=256,
                validation_split=0.25, verbose=1,
                callbacks = [es, checkpoint, tb_hist])  
        # 콜백에는 리스트 형태 

# 저장
model.save('./model/sample/cifar100/cifar100_model_save.h5')     # 훈련아래

model.save_weights('./model/sample/cifar100/cifar100_model_save_weight.h5')  # 훈련아래

# 4. 평가

loss_acc = model.evaluate(x_test,  y_test, batch_size= 64)

loss = hist.history['loss']  # model.fit 에서 나온 값
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc: ', acc)                               
print('val_acc: ', val_acc)
print('loss_acc: ', loss_acc)                     
                    

import matplotlib.pyplot as plt    

plt.figure(figsize = (10, 6))   # 10 x 6인치의 판이 생김

# 1번 그림
plt.subplot(2, 1, 1)  # (2, 1, 1) 2행 1열의 그림 1번째꺼 / subplot : 2장 그림               
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')                     
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')                  
plt.grid()                      # 격자 생성
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss','val_loss']) 
plt.legend(loc='upper right')   

# 2번 그림
plt.subplot(2, 1, 2)  # (2, 1, 2) 2행 1열의 그림 2번째꺼               
plt.plot(hist.history['acc'])                     
plt.plot(hist.history['val_acc'])                  
plt.grid()                         # 격자 생성
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()  

# acc:  [0.026746666, 0.035066668, 0.05072, 0.061573334, 0.07672, 0.083066665, 0.09261333, 0.09826667, 0.10653333, 0.110986665]
# val_acc:  [0.026399999856948853, 0.06431999802589417, 0.06272000074386597, 0.08879999816417694, 0.07624000310897827, 0.09831999987363815, 0.10903999954462051, 0.10440000146627426, 0.11400000005960464, 0.12600000202655792]
# loss_acc:  [3.8565898712158204, 0.12849999964237213]