import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                         
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train[0])                                         
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)       
print(y_test.shape)                                       # (10000,)



# 데이터 전처리 1. 원핫인코딩 : 당연하다             
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )                                                    
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                     


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape  = (28, 28, 1), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(80, (2, 2), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(60, (2, 2), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(40, (2, 2),padding = 'same'))
model.add(Conv2D(20, (2, 2),padding = 'same'))
model.add(Conv2D(10, (2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))                # 다중 분류

model.summary()


# """ model 저장 """
# model.save('./model/model_test01.h5')
# # 모델까지만 저장된다.(가중치는 저장이 안된다 / compile, fit 필요 )


# EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience = 20, mode= 'auto')
# Modelsheckpoint
modelpath = './model/check-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
                            verbose =1,
                            save_best_only= True, save_weights_only= False)
                                                 # 가중치만 저장하겠다 : False

#3. 훈련                      
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) # metrics=['accuracy']
hist = model.fit(x_train, y_train, epochs= 10, batch_size= 64, callbacks = [es, checkpoint],
                                   validation_split=0.2, verbose = 1)


""" model 저장 """
model.save('./model/model_test01.h5')
# fit한 다음에 저장하면 가중치가 저장된다.(compile, fit할 필요 X)


#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 64)

loss = hist.history['loss']                       
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc: ', acc)                               
print('val_acc: ', val_acc)
print('loss_acc: ', loss_acc)                     

import matplotlib.pyplot as plt    

plt.figure(figsize = (10, 6))                     

# 1번 그림
plt.subplot(2, 1, 1)                                            
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')                     
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')                  
plt.grid()                                        
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss','val_loss']) 
plt.legend(loc = 'upper right')                  
                                                 

# 2번 그림
plt.subplot(2, 1, 2)                                          
plt.plot(hist.history['acc'])                     
plt.plot(hist.history['val_acc'])                  
plt.grid()                                        
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()                                         

'''     
acc:  [0.90260416, 0.97216666, 0.9800208, 0.9836042, 0.98583335, 0.98716664, 0.9885833, 0.9897917, 0.9915625, 0.99135417]
val_acc:  [0.9559166431427002, 0.9783333539962769, 0.9790833592414856, 0.9825833439826965, 0.9833333492279053, 0.9810000061988831, 0.9793333411216736, 0.9857500195503235, 0.9850833415985107, 0.9835833311080933]
loss_acc:  [0.049524721492134265, 0.9851999878883362]
'''