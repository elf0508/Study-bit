# test0602_model1 복붙

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):  
        subset = seq[i : (i + size)]      
        aaa.append([item for item in subset])            
    print(type(aaa))
    return np.array(aaa)

size = 6 

# 1.데이터
# npy 불러오기

samsung = np.load('./data/samsung.npy', allow_pickle='True')
hite = np.load('./data/hite.npy', allow_pickle='True')

print(samsung.shape)  # (509, 1)
print(hite.shape)   # (509, 5)

samsung = samsung.reshape(samsung.shape[0], ) # (509, )

samsung = (split_x(samsung, size))
print(samsung.shape)   # (504, 6)  # 6일치씩 잘랐다.

x_sam = samsung[:, 0:5]  # 5일치 자른것까지 x로 하겠다.
y_sam = samsung[:, 5]

print(x_sam.shape)  # (504, 5) 앙상블할 때 행까지 맞춰줘야 함
print(y_sam.shape)  # (504, )

x_hit = hite[5:510, :]
print(x_hit.shape)  # (504, )

x_sam = x_sam.reshape(x_sam.shape[0], x_sam.shape[1], 1)

#2. 모델 구성: LSTM

input1 = Input(shape = (5, 1))

x1 = LSTM(70)(input1)
x1 = Dropout(0.1)(x1)
x1 = Dense(110)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(130)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(150)(x1)
x1 = Dropout(0.1)(x1)
x1 = Dense(170)(x1)


input2 = Input(shape = (5,)) 

x2 = Dense(50)(input2)
x2 = Dropout(0.1)(x2)
x2 = Dense(70)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(90)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(110)(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(130)(x2)


merge = concatenate([x1, x2])

middle = Dense(100)(merge)
middle = Dropout(0.1)(middle)
middle = Dense(80)(middle)
middle = Dropout(0.1)(middle)
middle = Dense(50)(middle)
middle = Dropout(0.1)(middle)
middle = Dense(30)(middle)
middle = Dropout(0.1)(middle)

output = Dense(1)(middle)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()

#3. 컴파일, 훈련

model.compile(loss ='mse', optimizer = 'adam')
model.fit([x_sam, x_hit], y_sam, epochs = 5)

print(x_sam[-1,:])
print(x_hit[-1, :])

