#  함수형 모델 : 앙상블
# 각 각의 데이터를 훈련 --> 엮어서 값이 나오도록
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

from keras.models import  Model
from keras.layers import Dense, Input

# 함수형에서는 Input, output이 무엇인지 명시를 해야한다
# input1 = Input(shape=(3, )) --> shape가 3인 이유는 데이터의 갯수가 3개이기 때문이다
# input1,input2  --> input이 2개인 이유는 x1, x2로 2개이기 때문이다
input1 = Input(shape=(3, ))
dense1_1 = Dense(10, activation='relu', name='bitking1')(input1)
dense1_2 = Dense(9, activation='relu',name='bitking2')(dense1_1)
dense1_3 = Dense(8, activation='relu',name='bitking3')(dense1_2)

input2 = Input(shape=(3, ))
dense2_1 = Dense(30, activation='relu')(input2)
dense2_2 = Dense(40, activation='relu')(dense2_1)
dense2_3 = Dense(50, activation='relu')(dense2_2)

# 위에서 만든 모델 2개를 엮어준다 : merge : 합병 / concatenate : 사슬처럼 엮다(단순 병합)
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3]) 
# x = x1, x2 로 2개이고, 오른쪽에서 왼쪽으로 나온 input값을 병합시켜준다 
# merge1 레이어를 만들어줌 (M1_끝 레이어 + M2_끝 레이어)

middle1 = Dense(30)(merge1) # 노드 30개짜리 만듬 / 딥러닝을 더 시킨다
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)
# merge 된 이후에 딥러닝이 구성
# output이 y1이므로 총 1개로 도출되어야 함


# 병합한 모델을 분리 시킨다 
##### output 모델 구성 ####


output1 = Dense(30)(middle1)  # y_M1의 가장 끝 레이어가 middle1
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

model.fit([x1_train, x2_train], y1_train, 
         epochs=30, batch_size=1,
        validation_split=0.25, verbose=1)  
        #  verbose 사용  0 : 빠르게 처리 할 때(시간 단축)
         

# 4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], y1_test, 
                     batch_size=1) 

                    
print("loss : ", loss)
print('전체 loss :', loss[0])

y1_predict = model.predict([x1_test, x2_test])

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
