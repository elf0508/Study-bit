# keras14_mlp 를 Sequential 에서 함수형으로 변경
# earlyStopping 적용 / 1 : 1

# 1. 데이터

import numpy as np 

# 1 ~ 100까지의 숫자
# x = np.array(range(1,101)) # 웨이트는 1, 바이어스는 100짜리

x1 = np.array([range(1,101), range(311,411), range(100)])  # (100, 3)
y1 = np.array([range(101,201),range(711,811),range(100)])

print(x1.shape)  
 # (3 , 100)

x1  = np.transpose(x1)
y1 = np.transpose(y1)

print(x1.shape)   
# (100, 3)

# 데이터 분리

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    # x, y, random_state=66, shuffle=True,
    x1, y1, shuffle=False,
    train_size=0.8
     # (80, 3)
)

print(x_train)
print(x_test)


# 2. 모델구성

from keras.models import Sequential, Model
from keras.layers import Dense, Input

# 함수형에서는 Input이 무엇인지 명시를 해야한다 / 3개 --> 3개
input1 = Input(shape=(3, ))
dense1_1 = Dense(10, activation='relu', name='bitking1')(input1)
dense1_2 = Dense(20, activation='relu',name='bitking2')(dense1_1)
dense1_3 = Dense(30, activation='relu',name='bitking3')(dense1_2)

output1 = Dense(3)(dense1_3)  # Dense( )는 열의 개수

model = Model(inputs = input1, 
            outputs = output1)

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.fit(x_train, y_train, epochs=210, batch_size=1,
            validation_split=0.25, verbose=1,
            callbacks=[early_stopping])
           

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 

print("loss : ", loss)

y_predict = model.predict(x_test)

print("=================")
print(y_predict)
print("=================")


# RMSE 구하기

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):

     return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))


# R2 구하기

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)

print("R2 : ", r2)


