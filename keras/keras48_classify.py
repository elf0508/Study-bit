# 이진분류 - O or X (1 or 0)
# 과제 : 최종 출력값을 0과 1이 나오도록 바꾸기

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

# 1.데이터  

x = np.array(range(1, 11))
y = np.array([1,0,1,0,1,0,1,0,1,0])

print("x.shape : ", x.shape)  # (10, )   스칼라가 10인 벡터 1개
print("y.shape : ", y.shape)  # (10, )


# 2. 모델구성

model=Sequential()

model.add(Dense(100, input_dim=1, activation='relu'))

model.add(Dense(50, activation='relu'))
model.add(Dense(20))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))  # sigmoid 를 곱해서 0 or 1 로 나오도록

model.summary()   

""" 
- 계산된 함수가 activation을 통해 다음 layer에 넘어간다.
- 가장 마지막 output layer값이 가중치와 '활성화 함수'와 곱해져서 반환된다. 
# sigmoid : 출력 값을 0과 1사이의 값으로 조정하여 반환한다.

# 손실 함수(Loss function)는 binary_crossentropy(이진 크로스 엔트로피)를 사용합니다.
# 이진분류 -> loss : binary_crossentropy(이진 크로스 엔트로피)
        #  output에 (Dense(1, input_dim=1, activation='sigmoid'))
"""

# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  # acc : 분류 모델 0 or 1

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x, y, 
         epochs=100, batch_size=1,
        validation_split=0.25, verbose=1,
        callbacks=[early_stopping])  
        # 콜백에는 리스트 형태 

# 4. 평가

loss, acc = model.evaluate(x, y, batch_size= 1)

print('loss :', loss )
print('acc :', acc )

# 4. 예측

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)

print('y_pred : ', y_pred)

# sigmoid 함수를 거치지 않은 걸로 보여짐

# y1_pred = np.where(y_pred >= 0.5, 1, 0)     
# print('y_pred :', y1_pred)
# """
# # np.where(조건, 조건에 맞을 때 값, 조건과 다를때 값)
# : 조건에 맞는 값을 특정 다른 값으로 변환하기
# """

# # y_predict = np.around(y_predict)

# # print('y_pred: ',y_predict)
# # y_predict.reshape(-1, 1)

# y_predict = y_predict.reshape(y_predict.shape[0])
# print(y_predict.shape)


# for i in range(len(y_predict)):
#     y_predict[i] = round(y_predict[i])

# print(y_predict)

# for i in range(len(y_pred)):
#     if y_pred[i]>0.5:
#         y_pred[i]=1
#     else:
#         y_pred[i]=0

y_pre=[int(round(i)) for i in y_predict]