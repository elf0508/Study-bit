# XOR모델 --> keras모델로(이진분류)

from sklearn.svm import LinearSVC    # 선형분류에 특화
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 군집 분석
                     #            분류                   회귀
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np

#1. 데이터
x_data = np.array([[0, 0],[1, 0],[0, 1],[1,1]])  # (4, 2)
y_data = np.array([0, 1, 1, 0])                  # (4, )

# x_data = [[0, 0],[1, 0],[0, 1],[1,1]]  # (4, 2)
# y_data = [0, 1, 1, 0]  

# x_data = np.array(x_data)
# y_data = np.array(y_data)

print(x_data.shape)
print(y_data.shape)

#2. 모델
# model = LinearSVC() 
# model = SVC()
# model = KNeighborsClassifier()  

model = Sequential()
model.add(Dense(10, input_shape =(2, ), activation='relu'))
# model.add(Dense(1, input_dim =(2, activation = 'sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(43, activation='relu'))
# model.add(Dense(54,activation='relu'))
# model.add(Dense(65, activation='relu'))
# model.add(Dense(86, activation='relu'))
# model.add(Dense(200, activation='relu'))  # acc : 0.75

model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))  # acc : 1.0

model.add(Dense(1, activation = 'sigmoid'))

model.summary()

#3. 훈련
# model.fit(x_data, y_data)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])  
model.fit(x_data, y_data, epochs =100, batch_size =1)


#4. 평가, 예측 / 주석처리 부분:머신러닝

# x_test = [[0, 0], [1, 0], [0, 1],[1, 1]]
# y_predict = model.predict(x_test)

                     
# acc = accuracy_score([0, 1, 1, 0], y_predict)  # evaluate = score()
#                      #  y_data

# print(x_test, '의 예측 결과: ', y_predict)
# print('acc = ', acc)
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과:  [0 1 1 0]    
# add =  1.0


x_test = np.array([[0, 0], [1, 0], [0, 1],[1, 1]])
     
loss, acc = model.evaluate(x_data, y_data, batch_size = 1)                    

y_predict = model.predict(x_test)
y_predict = np.where( y_predict > 0.5, 1, 0)

print(x_test, '의 예측 결과: ', y_predict)

print('loss = ', loss)
print('acc = ', acc)

