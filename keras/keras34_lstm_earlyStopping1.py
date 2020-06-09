# keras32_lstm_hamsu.py 을 복사 했음
# eariyStopping 모델로 바꾸기

from numpy import array
# import numpy as np
# x = np.array
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]]) 

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])     


print("x1.shape : ", x.shape)  # (13, 3)
print("y1.shape : ", y.shape)    #  (13, )

# x_predict = array([50, 60, 70])   # 아래쪽에 보기 쉽게 옮겨뒀다.
# x_predict = x_predict.reshape(1, 3, 1)

# x = x.reshape(13, 3, 1)  # (13, 3, 1)   

print("==== x reshape ====")
x = x.reshape(x.shape[0], x.shape[1], 1) 

# x의 shape 구조 
# model.fit 에서 batch_size를 자른다.
# x의shape = (batch_size, timesteps, feature)
#                행,         열,     몇개씩 자르는지
# input_shape = (timesteps, feature)
# input_length = timesteps, input_dim = feature

#  x.shape[0]=4, x.shape[1]=3, 1
# 마지막에 1을 추가==(13,3)에 있던 열작업을 1개씩 하겠다
# (13, 3, 1)의 값이 나온 .reshape는 전체를 곱해서, 똑같으면 맞는것이다.
#  13 * 3 = 13 * 3 * 1 

# print("=== x shape ===")
# print("x.shape : ", x.shape)
# print(x)


# 2. 모델구성
# 함수형에서는 Input, output 이 무엇인지 명시를 해야한다

model = Sequential()
input1 = Input(shape=(3, 1))

dense1 = LSTM(10, activation='relu')(input1)    # 각 layer의 이름을 명시해줘야 한다.
dense2 = Dense(17, activation='relu')(dense1)   # 각 layer마다 들어오는 input layer의 이름을 써줘야 한다.
dense3 = Dense(17, activation='relu')(dense2) 
dense4 = Dense(17, activation='relu')(dense3) 

output1 = Dense(1)(dense4) # y의 열이 1개여서 1이다.

model = Model(inputs=input1, outputs = output1)    # 함수형 Model을 가져다가 사용 할 건데, input레이어를 input1이라고 지정
                                                    # output레이어는 output1이라고 지정 하겠다.

model.summary()  


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x, y, 
         epochs=210, batch_size=1,
        validation_split=0.25, verbose=1,
        callbacks=[early_stopping])  
        # 콜백에는 리스트 형태  
                      
# 4. 평가, 예측

# x_predict = array([50, 60, 70])
# x_predict = x_predict.reshape(1, 3, 1) 을 정의 해야한다.

x_predict = array([50, 60, 70])
x_predict = x_predict.reshape(1, 3, 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

