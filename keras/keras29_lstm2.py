# 우리가 여태한 DNN(ANN) Deep, Artificial
# CNN Convolutional 복합적인 ex) 엄청나게 많은 이미지를 분석(쪼갤 때)
# RNN Recurrent 순환
# ex) 1 2 3 4 5 6 7 8 9 // ? 
# ex) 1만, 2만, ..., 5만 // ? 
# 시계열 time series

# RNN의 가장 대표적인 LSTM_Long Short(시계열) Term Memory
# 케글, 해커톤 대회에서 유행하는 프로그램이나 코딩법 유행이 달라짐
# 레거시한 방법(엄청 빠른 속도=레이어가 1개인 모델 속도) -> 텐서플로 -> 케라스 // 성능차이=0.99-->0.991
# RNN의 가장 주요한 3가지 타입=Simple RNN, GRU_한국인이 발명, LSTM(가장 좋은 성능)
# LSTM : 컬럼과 몇 개씩 잘라서 작업을 할 것 인가??
# 책 p.158 (그림 7-1)

from numpy import array
# import numba as np
# x = np.array
from keras.models import Sequential
from keras.layers import Dense, LSTM

'''
# 스칼라 벡터 행렬 텐서

# 스칼라=0차원 텐서. 하나의 숫자를 의미, ndim 0차원, rank=축(axis)
# 벡터=1D 텐서. 숫자의 배열, 딱 하나의 축
# 행렬=2D 텐서. 벡터의 배열, 2개의 축(행과 열). 숫자가 채워진 사각 격자
# 텐서=데이터를 위한 컨테이너. 거의 항상 수치형 데이터를 다루므로 숫자를 위한 컨테이너
# ex)행렬 / 다차원 numpy배열, 텐서에서는 차원=축(axis)
'''

# 1. 데이터

x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])  # (4, 3)
y = array([4,5,6,7])        # (4,  ) == 스칼라(4,5,6,7) 4개짜리 벡터 1개

print("x.shape : ", x.shape)  # (4, 3)
print("y.shape : ", y.shape)    # (4, )

'''
x    y
123  4
234  5
345  6
456  7
'''

# x = x.reshape(4, 3, 1)  # (4, 3, 1)  

 
'''
 (4, 3, 1)  == ( None, 3, 1)  
 행 무시, 열 우선
4행 3열을 1개씩 자르겠다. 
1개짜리는 무조건 '스칼라'로!!
2개 이상은 '행렬'로 해야 한다. 
'''

print("==== x reshape ====")

x = x.reshape(x.shape[0], x.shape[1], 1) 

# x의 shape 구조 
# model.fit 에서 batch_size를 자른다.
# x의shape = (batch_size, timesteps, feature)
#                행,         열,     몇개씩 자르는지
# input_shape = (timesteps, feature)
# input_length = timesteps, input_dim = feature

#  x.shape[0]=4, x.shape[1]=3, 1
# 마지막에 1을 추가==(4,3)에 있던 열작업을 1개씩 하겠다
# (4, 3, 1)의 값이 나온 .reshape는 전체를 곱해서, 똑같으면 맞는것이다.
#  4 * 3 = 4 * 3 * 1 

print("x.shape : ", x.shape)
print(x)

'''
x           y
[1][2][3]   4
......

'''


# 2. 모델구성

model = Sequential()

# model.add(LSTM(10, activation='relu', input_shape = (3, 1))) # 노드 10개 / Dense와 사용법 동일하나,input_shape=(열, 몇개씩 잘라서 작업한다.)
model.add(LSTM(10, input_length=3, input_dim=1))

model.add(Dense(3))  # 히든

model.add(Dense(1))  

model.summary()  


# 3. 훈련

model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=200, batch_size=1)

x_predict = array([5, 6, 7])
x_predict = x_predict.reshape(1, 3, 1)

# x_input = array([5, 6, 7])  # (3, ) <-- 스칼라 3짜리 이다.
# x_input = x_input.reshape(1, 3, 1)   # (1, 3, 1)  <-- 원래는 (3, ) 벡터였다.

'''
x_input  == x_test

# (3,) 스칼라 3개짜리 벡터1개로 x와 모양이 맞지 않음-->(1,3,1)로 reshape
# ( ,3,1) = 3개짜리 1개씩 작업하겠다. 그럼 행은 어떻게 정할까?
# x_input 3차원. 즉, 다 곱해보면 개수가 나옴. reshape 하기 전과 갯수가 같아야 함. 그래서 행은 1
'''

print(x_predict)  

'''
(1, 3, 1)  <-- 1행 3열을 1개씩 자르겠다.
[[[5]
  [6]
  [7]]] <-- (1, 3, 1)

  ==
  [[[5], [6], [7]]]
'''

yhat = model.predict(x_predict) 

print(yhat) # [[7.9042625]]   <-- 할 때 마다 값이 바뀐다.


# yhat의 출력값이 왜 하나죠?
# |---x---|--y--|
# |1  2  3|  4  |
# |2  3  4|  5  |
# |3  4  5|  6  |
# |4  5  6|  7  |
# |5  6  7|  ?? |   # model.predict 구간에서 예측되는 y값은 1개


# 4. 예측

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)