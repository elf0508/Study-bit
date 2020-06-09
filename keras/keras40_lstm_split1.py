# keras40_lstm_split1.py

import numpy as np
from keras.models import Sequential #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, LSTM # Dense와 LSTM 레이어를 쓰겠다


# 1. 데이터 생성
a = np.array(range(1, 11))        # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (10, )
size = 5     # time_steps = 4     #      0  1  2  3  4  5  6  7  8  9

# 실습 : LSTM 모델을 완성하시오. 
# keras39_split.py 모델 사용

def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa)) 
    return np.array(aaa)
dataset = split_x(a, size)

# 6바이 5로 나뉜다
dataset = split_x(a, size)  # (6, 5)
print(dataset)
print(dataset.shape)
print(type(dataset))   # <class 'numpy.ndarray'>

x = dataset[:, 0:4]   # [:] : 전부 다 -> 6에 대한 모든 행 / 열 : [0:4] - 0부터 4까지 => (0,1,2,3)
                        # 모든 행을 가져오는데, (0,1,2,3)까지 가져오겠다.
                        #  [[ 1  2  3  4 ]
                        #  [  2  3  4  5 ]
                        #  [  3  4  5  6 ]
                        #  [  4  5  6  7 ]
                        #  [  5  6  7  8 ]
                        #  [  6  7  8  9 ]]  를  나타낸다.

y = dataset[:, 4]
#  [[ 5]
#  [  6]
#  [  7]
#  [  8]
#  [  9]
#  [ 10]] 를 나타낸다.

x = x.reshape(x.shape[0],x.shape[1],1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

# x = np.reshape(x, (6, 4, 1))
# # 전체 행 : 6개 / 컬럼: 4개 
# # x2 = x.reshape(6, 4, 1)
# print(x.shape)
# print(x.shape)

# 2. 모델구성
model = Sequential()
model.add(LSTM(10, input_shape=(4, 1)))
model.add(Dense(5))
model.add(Dense(640))
model.add(Dense(16))  
model.add(Dense(1))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=10000, batch_size=3, callbacks=[early_stopping])  

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test) 
print('mse 는',mse)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)


'''
x = []
y = []
for i in dataset :
    y.append(i[-1])
for i in dataset :
    x.append(list(i[:4]))
print(x) 
print(y)

x = np.array(x)
y = np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense 
model = Sequential()
model.add(Dense(5, input_dim=4)) 
model.add(Dense(16)) 
model.add(Dense(32))  
model.add(Dense(1)) 

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=500, batch_size=3)  

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test) 
print('mse 는',mse)

# 예측값 y햇 만들어주기
y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print('MSE 는', mean_squared_error(y_test, y_predict) )
print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)
'''