# 85번 복붙

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist   # keras에서 제공되는 예제 파일 

mnist.load_data()          # mnist파일 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist에서 이미 x_train, y_train으로 나눠져 있는 값 가져오기

print(x_train[0])    # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train: ' , y_train[0])    # 5

print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)     # (10000, 28, 28)
print(y_train.shape)    # (60000,)   : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)     # (10000,)



# 데이터 전처리 1. 원핫인코딩 : 당연하다  => y 값  
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )            => x 값                                           
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                     


# 불러오기
from keras.models import load_model 

model = load_model('./model/model_test01.h5')

model.summary()


#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 64)

print('loss_acc: ', loss_acc)                     

y_predict = model.predict(x_test[0: 10])
y_predict = np.argmax(y_predict, axis =1)
print(y_predict)


# loss = hist.history['loss']    # model.fit 에서 나온 값
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

# print('acc: ', acc)                               
# print('val_acc: ', val_acc)
# print('loss_acc: ', loss_acc)                     

# import matplotlib.pyplot as plt    

# plt.figure(figsize = (10, 6))  # 10 x 6인치의 판이 생김

# # 1번 그림
# plt.subplot(2, 1, 1)  # (2, 1, 1) 2행 1열의 그림 1번째꺼 / subplot : 2장 그림               
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')                     
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')                  
# plt.grid()   # 격자 생성
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# # plt.legend(['loss','val_loss']) 
# plt.legend(loc='upper right')   

# # 2번 그림
# plt.subplot(2, 1, 2)   # (2, 1, 2) 2행 1열의 그림 2번째꺼               
# plt.plot(hist.history['acc'])                     
# plt.plot(hist.history['val_acc'])                  
# plt.grid()         # 격자 생성
# plt.title('accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['acc','val_acc'])

# plt.show()                                         

