# 1.데이터
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

# 2.모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(1, ))
 
x1 = Dense(100)(input1)    # 히든 레이어
x1 = Dense(100)(x1)       
x1 = Dense(100)(x1)

x2 = Dense(50)(x1)       # 분리
output1 = Dense(1)(x2)   # dense_5

x3 = Dense(70)(x1)
x3 = Dense(70)(x3)

output2 = Dense(1, activation ='sigmoid')(x3)   # dense_8

model = Model(inputs = input1, outputs = [output1, output2])

model.summary()

# 3.컴파일, 훈련
model.compile(loss = ['mse', 'binary_crossentropy'], optimizer ='adam',
                metrics = ['mse', 'acc'])

model.fit(x_train, [y1_train, y2_train], epochs = 100, batch_size = 1)

# 4.평가, 예측
loss = model.evaluate(x_train, [y1_train, y2_train])

print("loss : ", loss)

x1_pred = np.array([11,12,13,14])

y_pred = model.predict(x1_pred)

print(y_pred)

# loss :  [0.7809974551200867,    <-- 전체 loss값
#          0.10150198638439178,   <-- dense_5
#          0.6794954538345337,    <-- dense_8
#          0.10150198638439178,   <-- dense_5
#          0.8999999761581421,    <-- dense_5
#          0.2432716190814972,    <-- dense_8
#          0.6000000238418579]    <-- dense_8

# 회귀
# [array([[10.452218],
#        [11.40505 ],
#        [12.357881],
#        [13.310711]], dtype=float32)

# 분류
# [0.3867871 ],
#        [0.36735195],
#        [0.34833854],
#        [0.3297963 ]], dtype=float32)] 






