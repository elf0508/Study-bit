# (?, 13) 13, 1, 1

# PCA

import numpy as np
from sklearn.datasets import load_boston
from keras.layers import LSTM, Dense,Conv2D, Flatten,MaxPooling2D
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset=load_boston()
x=dataset.data
y=dataset.target

from sklearn.model_selection import train_test_split


#pca해주기 전에 표준화
x=StandardScaler().fit_transform(x)

pca=PCA(n_components=2) #주성분 개수
trans_x=pca.fit_transform(x)

print("trans_x:",trans_x)
print("trans_x.shape:",trans_x.shape)


x_train,x_test,y_train,y_test=train_test_split(trans_x,y,train_size=0.8)
print(x_train.shape)
print(x_test.shape)

x_train=x_train.reshape(404,2,1,1)
x_test=x_test.reshape(102,2,1,1)


print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)


# 모델 구성

model=Sequential()

model.add(Conv2D(10, (1,1), input_shape=(2,1,1), activation='relu'))  
#shape보다 kernel_size는 작게 잘라주어야 한다. 

model.add(Conv2D(20,(1,1), activation='relu'))
model.add(Conv2D(10,(1,1), activation='relu'))
model.add(Conv2D(15,(1,1), activation='relu'))

model.add(Flatten())

model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='aut')

model.fit(x_train,y_train, epochs = 10, batch_size = 1)


loss_acc = model.evaluate(x_test, y_test, batch_size = 1)

y_predict = model.predict(x_test)

print("y_predict:", y_predict)

from sklearn.metrics import mean_squared_error as mse

def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_predict,y_test))

print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

''' 
RMSE: 6.929050015911096
R2: 0.4993894659743946
'''