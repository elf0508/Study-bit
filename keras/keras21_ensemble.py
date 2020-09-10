#  함수형 모델 : 앙상블 
# 앙상블 : 음악 합주 / 어마한 데이터 셋이 2개 이상을 합쳐보자
# 각 각의 데이터를 훈련 --> 엮어서 값이 나오도록

# 데이터

import numpy as np 

x1 = np.array([range(1,101), range(311,411), range(100)])  # 3행 100열
y1 = np.array([range(711,811), range(711, 811), range(100)])

# Q. y1에서 range값이 중복되어도 되나요?
# A. 상관없지. 위아래 세트로 가중치가 나오잖아.
# 1st w=1, b=700 / 2nd w=1, b=400 / 3rd w=1, b=0

x2 = np.array([range(101,201), range(411,511), range(100,200)])  
y2 = np.array([range(501,601), range(711, 811), range(100)])

# x = np.transpose(x)  # 100행 3열로 바뀌었다

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)


# 데이터 분리  / _size의 이름을 같도록 해보자

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(
#     x, y, random_state=66, shuffle=True,
    x1, y1, shuffle=False,
    train_size=0.8  
)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    # x, y, random_state=66, shuffle=True,
    x2, y2, shuffle=False,
    train_size=0.8
)


# 2. 모델구성

from keras.models import Sequential, Model
from keras.layers import Dense, Input  # 함수형 모델은 input, output을 명시해줘야함


# 모델 2개 만듬
input1 = Input(shape=(3, ))
dense1_1 = Dense(10, activation='relu', name='bitking1')(input1)
dense1_2 = Dense(9, activation='relu',name='bitking2')(dense1_1)
dense1_3 = Dense(8, activation='relu',name='bitking3')(dense1_2)

input2 = Input(shape=(3, ))
dense2_1 = Dense(40, activation='relu')(input2)
dense2_2 = Dense(50, activation='relu')(dense2_1)
dense2_3 = Dense(50, activation='relu')(dense2_2)

# 위에서 만든 모델 2개를 엮어준다 : merge : 합병 / concatenate : 사슬처럼 엮다(단순 병합)
# M1, M2 두 개 모델의 레이어들을 엮어주는 API 호출
from keras.layers.merge import concatenate

merge1 = concatenate([dense1_3, dense2_3]) 
# 2개 이상은 리스트 : [ ]로 묶어줘야한다
# merge1 레이어를 만들어줌 (M1_끝 레이어 + M2_끝 레이어)

middle1 = Dense(30)(merge1) # 노드 30개짜리 만듬 / 딥러닝을 더 시킨다
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)
# merge 된 이후에 딥러닝이 구성
# output이 y1, y2이므로 아웃풋이 총 2개로 도출되어야 함

# 병합한 모델을 분리 시킨다
##### output 모델 구성 ####

output1 = Dense(30)(middle1)  # M1의 가장 끝 레이어가 middle1
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(30)(middle1)   # M2의 가장 끝 레이어가 middle1
output2_2 = Dense(7)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs=[input1, input2], outputs = [output1_3, output2_3])
# 함수형 모델은 범위가 어디서부터 어디까지인지 명시. 히든레이어는 명시해줄 필요 없으므로 input과 output만 명시

model.summary()

# M1-M2가 번갈아 가면서 훈련될 예정
# model.summary()의 layer 이름 변경하는 파라미터? ==> name 파라미터


# 3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=30, batch_size=1,
         validation_split=0.25, verbose=1)  
        #  verbose 사용  0 : 빠르게 처리 할 때(시간 단축)
  # list로 묶어서 한번에 model.fit 완성
         

# 4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1) 

print("loss : ", loss)
print('전체 loss :', loss[0])
print('모델1의 loss :', loss[1])
print('모델2의 loss :', loss[2])

y1_predict,y2_predict = model.predict([x1_test, x2_test])

#(20,3)짜리 2개 왜? train_size=0.8이기 때문에

print("=================")
print(y1_predict)
print("=================")
print(y2_predict)
print("=================")


# RMSE 구하기

from sklearn.metrics import mean_squared_error

def RMSE(y1_test, y1_predict):

     return np.sqrt(mean_squared_error(y1_test, y1_predict))
     # sklearn에서는 list를 감당할 수 없다
     
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)


# R2 구하기

from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)

print("R2_1 : ", r2_1)
print("R2_1 : ", r2_2)
print("R2 : ", (r2_1 + r2_2)/2)

