# 1.데이터
import numpy as np

x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])

y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

# 2.모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1, ))
 
x1 = Dense(100)(input1)    # 히든 레이어
x1 = Dense(100)(x1)       
x1 = Dense(100)(x1)

input2 = Input(shape=(1, ))
 
x2 = Dense(100)(input2)    # 히든 레이어
x2 = Dense(100)(x2)       
x2 = Dense(100)(x2)

merge = concatenate([x1, x2])

x3 = Dense(100)(merge)

output1 = Dense(1)(x3)

x4 = Dense(70)(merge)
x4 = Dense(70)(x4)

output2 = Dense(1, activation ='sigmoid')(x4)   # dense_8

model = Model(inputs = [input1, input2], outputs = [output1, output2])

model.summary()

# 3.컴파일, 훈련
model.compile(loss = ['mse', 'binary_crossentropy'], 
                optimizer='adam',
                loss_weights=[0.1, 0.9],   # loss를 분류에서 0.9 주겠다.
                metrics=['mse', 'acc'])

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1)

# 4.평가, 예측
loss = model.evaluate([x1_train, x2_train], [y1_train, y2_train])

print("loss : ", loss)

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])

y_pred = model.predict([x1_pred, x2_pred])

print(y_pred)

# loss :  [0.6135311126708984, 
#          0.0034314352087676525, 
#          0.6813200116157532, 
#          0.0034314352087676525, 
#          1.0, 
#          0.24413907527923584,
#          0.6000000238418579] 

# 회귀

# [array([[11.0858965],
#        [12.091235 ],
#        [13.096573 ],
#        [14.10191  ]], dtype=float32)

# 분류

# array([[0.41809458],
#        [0.40233135],
#        [0.38676745],
#        [0.37143135]], dtype=float32)]




