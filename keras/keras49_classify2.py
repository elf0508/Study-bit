# 다중분류 

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
from keras import  optimizers


# 1.데이터  

x = np.array(range(1, 11))      # (10, )  <-- 스칼라가 10인 벡터 1개
y = np.array([1,2,3,4,5,1,2,3,4,5])   # (10, )  <-- 스칼라가 10 

print("x.shape : ", x.shape)  # (10, )   <-- 스칼라가 10인 벡터 1개
print("y.shape : ", y.shape)  # (10, )

# 1차원 데이터를 2차원으로 바꿔줘야한다.
 
from keras.utils import np_utils

y = np_utils.to_categorical(y)   # y값을 분류해서,  다시 y에게 반환한다.

print(y)

# 원핫 인코딩

""" 
- 다중분류 모델은 반드시 one_hot_encoding사용
- 다중 클래스 분류 문제가 각 클래스 간의 관계가 균등하기 때문에
  ex) y가 1 과 5로 분류된다면 5에 값이 치중된다.

- 해당 숫자에 해당되는 자리만 1이고 나머지는 0으로 채운다. 
- '0'부터 인덱스가 시작이다.
  \ 자리 : 0   1   2   3   4   5  
  숫자 ---------------------------
       1 : 0   1   0   0   0   0
       2 : 0   0   1   0   0   0
       3 : 0   0   0   1   0   0
       4 : 0   0   0   0   1   0
       5 : 0   0   0   0   0   1  
"""

print(y)                    
print(y.shape)                     # (10, 6)  : 자동으로 0을 인식해서 6개(dimension이 늘어남)

y = y[:, 1:]                       # 우리는 column이 5개(원래 dimension)임으로 reshape해준다.
print(y.shape)                     # (10, 5) 


# 2. 모델구성

model=Sequential()

model.add(Dense(100, input_dim=1, activation='relu'))

model.add(Dense(50, activation='relu'))
model.add(Dense(20))
model.add(Dense(10, activation='relu'))

model.add(Dense(5, activation='softmax'))  # (10, 5)

# softmax 
# : 다중 분류의 시그모이드 결과 값인 '점수'들을 총 합이 1인 '확률'로 수치화하기 위해 사용하는 함수
# : 출력 확률 범위. 범위는 0에서 1이며, 모든 확률의 합은 1과 같다.
#  다중 분류는 'softmax' 사용
#  : 가장 큰 수 빼고는 전부 0으로 나옴

model.summary()   

# 3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#  loss = 'categorical_crossentropy' : 다중분류에서 사용 

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
y_pred = model.predict(x_pred)   # 'softmax'가 적용되지 않은 모습으로 나옴

print('y_pred : ', y_pred)  # 1,2,3,4,5 중에서 값이 나와야 한다.

print(y_pred.shape)                               # (3, 5)

""" x 하나 집어 넣으면 [ 5 ]개가 나옴 (one_hot_encoding때문)
   [ 1   2   3   4   5 ]
y1 : 1   0   0   0   0    => 1
y2 : 0   1   0   0   0    => 2
y3 : 0   0   1   0   0    => 3
"""


"""
one_hot_decoder
: np.argmax()
: 최대값의 색인 위치를 찾는다.
"""
#1. 함수 사용

def decode(datum):
    return np.argmax(datum)
  
for i in range(y_pred.shape[0]):                   # y_pred.shape[0] = 3, i = [0, 1, 2]                     
    y2_pred = decode( y_pred[i])       
    print('y2_pred:', y2_pred + 1)

#2.
y3_pred = np.argmax(y_pred, axis= 1) + 1           # 뒤로 한자리씩 넘겨준다.
print(y3_pred)

# loss,acc = model.evaluate(x, y, batch_size=1)
# print("loss : ", loss)
# print("acc : ", acc)

# x_pred = np.array([1,2,3,4,5]) #output이 6개라서 하나 넣으면 6개 나와
# y_predict = model.predict(x_pred)
# y_predict = np.argmax(y_predict,axis=1) + 1
# print('y_pred: ',y_predict) 

#!!!!!!y_pred를 바꾸는 함수????!!
# y_pred:  [[0.08447615 0.18838052 0.18655092 0.18472885 0.16989201 0.18597163] 제일 큰 값  1
                # 0        1             2           3          4          5
#  [0.05172801 0.17869125 0.19075894 0.19404957 0.17365383 0.2111184 ]                     5
#  [0.03112592 0.16643532 0.19148125 0.20050484 0.17465068 0.23580205]]     
# 
# y_pred:  [[0.21092378 0.20677215 0.19483757 0.19502336 0.19244318] 1
#  [0.20905657 0.20650439 0.19460683 0.19570489 0.19412737] 1
#  [0.2025705  0.20501168 0.19430959 0.19903412 0.19907413]] 2

# y_pred:  [[0.21215552 0.20070623 0.2002933  0.19051124 0.19633374] 1
#  [0.19849712 0.19591473 0.20413128 0.19742124 0.20403571] 3
#  [0.18677607 0.19071889 0.20708707 0.20335016 0.21206777]] 5