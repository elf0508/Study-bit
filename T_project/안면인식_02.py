import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,BatchNormalization,GlobalAveragePooling2D,Activation,Reshape
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from ml_metrics import rmse


#Load-------------------------------------------------------------------------------------------------------------------

a = np.load('D:/Study-bit/T_project/H_Resolution/koreaface/xe.npy')

# a=np.load('D:/s/koreaface/xb.npy')
# a=np.load('D:/s/koreaface/xc.npy')
# a=np.load('D:/s/koreaface/xd.npy')
## a=np.load('D:/s/koreaface/xe.npy')
# a=np.load('D:/s/koreaface/xf.npy')
# a=np.load('D:/s/koreaface/xg.npy')
# a=np.load('D:/s/koreaface/xh.npy')
# a=np.load('D:/s/koreaface/xi.npy')
# a=np.load('D:/s/koreaface/xj.npy')
# a=np.load('D:/s/koreaface/xk.npy')
# a=np.load('D:/s/koreaface/xl.npy')
# a=np.load('D:/s/koreaface/xn.npy')
# a=np.load('D:/s/koreaface/xm.npy')
# a=np.load('D:/s/koreaface/xo.npy')
# a=np.load('D:/s/koreaface/xp.npy')
# a=np.load('D:/s/koreaface/xq.npy')
# a=np.load('D:/s/koreaface/xr.npy')

#-----------------------------------------------------------------------------------------------------------------------
#한 사람당 이미지 장수는 ? 10800개
#사이즈가 너무 커서 메모리 용량 부족 한개씩 가져와야된다.

#-----------------------------------------------------------------------------------------------------------------------

print(a.shape)   # (540, 20, 185, 270, 3)

a = a.reshape(10800, 185, 270, 3)
# a = a.reshape(10800, int(a.shape[0]*a.shape[1]/10800), 100, 143, 3)

# 파일당 인원 체크
# 1.=22 , 2.=21 3.=18 4.=20 5.=?(스크래치),6.=20 ,7=24 ,8.=20,9.=23,10.=20 11.=23 12.=21,13.=21,14=22,15=31,16.=25,17.=24,18.=23
# 총 인원 =334

print(a.shape)   # (10800, 185, 270, 3)

k = np.arange(0, int(a.shape[1]),1) 

print(k.shape)   # (185, )

k = list(k)

u = int(a.shape[1])  #22
p = int(a.shape[0])


c = []
for i in tqdm(k):
    for j in range(p):
        c.append(i)

y = np.asarray(c).reshape(-1,1)

print(y.shape)  # (1998000, 1)

####################################

x, x_pred, y, y_hat = train_test_split(a, y, train_size = 0.9)

X = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])

print(X.shape)

X_pred = x_pred.reshape(x_pred.shape[0]*x_pred.shape[1], x_pred.shape[2], x_pred.shape[3], x_pred.shape[4])

print(X_pred.shape)


# 예측데이터를 10% 떼어낸다. Train에는 전체데이터 90%로 사용한다.
# # X_pred = 1080개
# # X = 9720개

## list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# # list=tf.keras.utils.to_categorical(k)

#Categorical Method-------------------------------------------------------------------

# u=int(x.shape[1]) #22
# output_size=int(x.shape[1])
# p=int(x.shape[0])
# list=tf.keras.utils.to_categorical(k,334)
#
# # print(list.shape)
# for i in tqdm(range(u)):
#     for j in range(p):
#         y.append((list[i,:]))
# y=np.asarray(y)
#Sparse Categorical Method-------------------------------------------------------------

output_size = int(x.shape[1])

#  [None,100,143,3]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)
#
# print(x_train.shape,x_test.shape)
# print(y_train.shape,y_test.shape)

#Model-Identification----------------------------------------------------------------------------------------------------------------

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,4),padding='valid',input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(2,3),padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(1,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# model.add=Dense(512,activation='relu')

model.add(Dense(output_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# #Train_and_Pred-------------------------------------------------------------------------------------------------------
from keras.models import load_model

##first_ep---------------------------------------------------------------------------------------------------------------

model.fit(x_train, y_train,epochs=100, validation_split=(0.2))

model.evaluate(x_test, y_test)

model.save('./한안.h5')

#ep---------------------------------------------------------------------------------------------------------------------
# model = load_model('./한안.h5')
# model.fit(x_train,y_train,epochs=100,validation_split=(0.2))
# model.evaluate(x_test,y_test)
# model.save('./한안.h5')

#predict----------------------------------------------------------------------------------------------------------------

# pred=model.predict(X_pred)


