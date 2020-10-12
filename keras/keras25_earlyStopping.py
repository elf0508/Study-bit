#  함수형 모델 : 앙상블
# R2의 값이 올라가다가 갑자기 흔들리는 지점이 생김
# RMSE는 값이 내려가다가 갑자기 수치가 흔들리는 지점이 생김
# 저 지점이 오기 전에 끊을 수 있을까? 최고점을 잡기는 힘들다
# 최고점을 조금 지난 상태에서 epochs를 확인하여 끊을 수 있다. 그것이 earlyStopping

# 2개가 들어가서 1개가 나오는 것

# 데이터

import numpy as np 

x1 = np.array([range(1,101), range(311,411), range(411,511)])  # 100바이 3  (100, 3)
x2 = np.array([range(711,811), range(711, 811),range(511, 611)])

y1 = np.array([range(101,201), range(411,511), range(100)])  

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

# 데이터 분리  / _size의 이름을 같도록 해보자

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(
#     x, y, random_state=66, shuffle=True,
    x1, y1, shuffle=False,
    train_size=0.8  
)

x2_train, x2_test = train_test_split(
    # x, y, random_state=66, shuffle=True,
    x2, shuffle=False,
    train_size=0.8
)

# 2. 모델구성

from keras.models import Sequential, Model
from keras.layers import Dense, Input

# 함수형에서는 Input, output이 무엇인지 명시를 해야한다
# input1 = Input(shape=(3, )) --> shape가 3인 이유는 데이터의 갯수가 3개이기 때문이다
# input1,input2  --> input이 2개인 이유는 x1, x2로 2개이기 때문이다
input1 = Input(shape=(3, ))
dense1_1 = Dense(15, activation='relu', name='bitking1')(input1)
dense1_2 = Dense(17, activation='relu',name='bitking2')(dense1_1)
dense1_3 = Dense(20, activation='relu',name='bitking3')(dense1_2)

input2 = Input(shape=(3, ))
dense2_1 = Dense(25, activation='relu')(input2)
dense2_2 = Dense(20, activation='relu')(dense2_1)
dense2_3 = Dense(30, activation='relu')(dense2_2)

# 위에서 만든 모델 2개를 엮어준다 : merge : 합병 / concatenate : 사슬처럼 엮다(단순 병합)
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3]) 
# x = x1, x2 로 2개이고, 오른쪽에서 왼쪽으로 나온 input값을 병합시켜준다 

middle1 = Dense(30)(merge1) # 노드 30개짜리 만듬 / 딥러닝을 더 시킨다
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

# 병합한 모델을 분리 시킨다 

##### output 모델 구성 ####

output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)
# output1_3 = Dense(3)(output1_2) --> Dense(3)은 데이터가 3개이기 때문이다

model = Model(inputs=[input1, input2], 
            outputs = output1_3)
# inputs=[input1, input2] --> x값이 x1, x2로 2개 이상이기때문에 리스트 [ ]로 묶어야한다
# outputs = output1_3  --> y값이 1개이기 때문에

model.summary()


# 3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# earlyStopping 파라미터를 호출
# 카멜케이스 형태로 앞의 'E'가 대문자

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
# monitor=모니터링 지표
# patience=선생님의 개그를 견디는 횟수 / mode가 min이면 그 횟수 이하, max면 그 횟수 이상
# mode=min, max, auto

model.fit([x1_train, x2_train], y1_train, 
         epochs=210, batch_size=1,
        validation_split=0.25, verbose=1,
        callbacks=[early_stopping])  
        # 콜백에는 리스트 형태  
         

# 4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], y1_test, 
                     batch_size=1) 

# y1_output1에 대한 loss(1)
# y1_output1에 대한 mse(1)
# 총 2개의 반환값
                    
print("loss : ", loss)
print('전체 loss :', loss[0])

y1_predict = model.predict([x1_test, x2_test])
#(20,3)짜리 3개 왜? train_size=0.8이기 때문에

print("=================")
print(y1_predict)
print("=================")


# RMSE 구하기

from sklearn.metrics import mean_squared_error

def RMSE(y1_test, y1_predict):

     return np.sqrt(mean_squared_error(y1_test, y1_predict))

RMSE1 = RMSE(y1_test, y1_predict)

print("RMSE1 : ", RMSE1)


# R2 구하기

from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y1_predict)

print("R2_1 : ", r2_1)
