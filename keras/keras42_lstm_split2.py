# keras40_lstm_split1.py
# 실습 1. train, test 분리할것. (8:2) 
# 실습 2. 마지막 6개의 행을 predict로 만들고 싶다.
# 실습 3. validation을 넣을 것 (train 20% 사용하기)

import numpy as np
from keras.models import Sequential #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, LSTM # Dense와 LSTM 레이어를 쓰겠다


# 1. 데이터 생성

a = np.array(range(1, 101))        # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10...100] (100, )
size = 5     # time_steps = 4      #      0  1  2  3  4  5  6  7  8  9 ...


def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :   
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa)) 
    return np.array(aaa)

# 96바이 5로 나뉜다
dataset = split_x(a, size)  # (96, 5)
x = np.array(dataset[:,:4])
y = np.array(dataset[:,4])
x_predict = x[-6:,:]
x_predict_y = y[-6:]

x = x.reshape(x.shape[0],x.shape[1],1)
x_predict = x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2, shuffle=False
)


#2. 모델구성

from keras.models import Sequential 
from keras.layers import LSTM, Dense 

model = Sequential()

model.add(LSTM(320, input_shape=(4,1))) # input을 넣는거야 무조권 ^_^ 

model.add(Dense(32))
model.add(Dense(640))
model.add(Dense(320))
model.add(Dense(16))  

model.add(Dense(1)) 


#3. 설명한 후 훈련

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') 

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train,y_train, epochs=50, batch_size=5, callbacks=[early_stopping], validation_split=0.2)  


#4. 평가와 예측

loss,mse = model.evaluate(x_test,y_test) 
print('mse 는',mse)

x_predict = model.predict(x_predict)
print(x_predict)

from sklearn.metrics import mean_squared_error 

def RMSE(y_test, y_predict) :

    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE 는', RMSE(x_predict_y, x_predict) )


# R2 구하기

from sklearn.metrics import r2_score 

r2 = r2_score(x_predict_y, x_predict)

print('R2는 ', r2)