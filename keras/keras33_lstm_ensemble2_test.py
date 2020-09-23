# keras29_lstm3_scale.py
# 앙상블 모델로 바꾸기

from numpy import array
# import numba as np
# x = np.array


# 1. 데이터

x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]]) 

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]]) 

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])   

print("x.shape : ", x1.shape)  # (13, 3)  
print("y.shape : ", y.shape)    #  (13, )

# x = x.reshape(13, 3, 1)  # (13, 3, 1)   

print("==== x reshape ====")
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) 
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) 
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

print("=== x1 shape ===")
print("x1.shape : ", x1.shape)
print(x1)

print("=== x2 shape ===")
print("x2.shape : ", x2.shape)
print(x2)


# 2. 모델구성

from keras.models import Model
from keras.layers import Dense, LSTM, Input

####### input 1  #######
input1 = Input(shape=(3,1 ))
input1_1 = LSTM(10)(input1)
input1_2 = Dense(10)(input1_1)
input1_3 = Dense(10)(input1_2)

####### input 2  #######
input2 = Input(shape=(3,1 ))
input2_1 = LSTM(10)(input2)
input2_2 = Dense(10)(input2_1)
input2_3 = Dense(10)(input2_2)

####  합병 ######
from keras.layers.merge import concatenate
merge1 = concatenate([input1_3, input2_3]) 

middle1 = Dense(30)(merge1) # 노드 30개짜리 만듬 / 딥러닝을 더 시킨다
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

# 병합한 모델을 분리 시킨다 
##### output 모델 구성 ####

output1 = Dense(30)(middle1)  # y_M1의 가장 끝 레이어가 middle1
output1_2 = Dense(7)(output1)
output1_3 = Dense(1)(output1_2)  # y의 열에 맞춰야한다.

### 함수형 모델 명시 ###   Sequential은 맨 위에 명시 해야한다.  model = Sequential()
model = Model(inputs=[input1, input2], 
            outputs = output1_3)

model.summary()  

# 3. 훈련

model.compile(optimizer='adam', loss='mse')

model.fit([x1, x2], y, epochs=850, batch_size=32)  


# 4. 예측

x1_predict = array([55,65,75])     
x2_predict = array([65,75,85])


x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

print(x1_predict)
print(x2_predict)

y_predict = model.predict([x1_predict,x2_predict])   # fit의 x 모양,개수 / y의 개수를 맞춰준다.

print(y_predict)
