# 함수형 모델
# 함수는 재사용을 위해 생성
# A_sequential 모델 + B_sequential 모델. 어떻게 한번에 묶을까?
# A_sequential 모델==A 함수, B_sequential 모델==B함수로 묶어서 전체를 또다른 하나의 신경망으로 표현

# 데이터
import numpy as np 

# 1 ~ 100까지의 숫자
# x = np.array(range(1,101)) # 웨이트는 1, 바이어스는 100짜리

x = np.transpose([range(1,101), range(311,411), range(100)])  # 100행 3열로 바뀌었다
y = np.transpose(range(711,811))

# np = np.transpose(x)
# np = np.transpose(y)

print(x.shape)
print(y.shape)

# 데이터 분리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    # x, y, random_state=66, shuffle=True,
    x, y, shuffle=False,
    train_size=0.8 
)

print(x_train)
# print(x_val)
print(x_test)


# 2. 모델구성
# 함수형에서는 Input, output 이 무엇인지 명시를 해야한다
from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))  #행무시 열우선. 100행 3열이기에 3열 표시
# 변수명은 소문자로 암묵적인 룰
# 함수형 모델에서는 keras.layer의 계층 친구인 Input을 명시해줘야함
dense1 = Dense(15, activation='relu')(input1)
# activation=활성화 함수 # 앞단의 아웃풋이 뒤 꽁지에 붙음
dense2 = Dense(18, activation='relu')(dense1)
dense3= Dense(19, activation='relu')(dense2)
dense4= Dense(21, activation='relu')(dense3)
dense5 = Dense(33, activation='relu')(dense4)   #activation에도 디폴트가 있음
output1 = Dense(1)(dense5)

# 이름이 같아도 동작한다
# input1 = Input(shape=(3, ))
# dense1 = Dense(15, activation='relu')(input1)
# dense1 = Dense(18, activation='relu')(dense1)
# dense1 = Dense(19, activation='relu')(dense1)
# dense1 = Dense(21, activation='relu')(dense1)
# dense1 = Dense(33, activation='relu')(dense1)
# output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs = output1)

# 순차적 모델은 model = Sequential()이라고 명시를 하고 시작했지만,
# 함수형 모델은 범위가 어디서부터 어디까지인지 명시해줘야 함. 히든레이어는 명시해줄 필요 없으므로 input과 output만 명시

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=30, batch_size=1,
           validation_split=0.25, verbose=0)  
        #  verbose 사용  0 : 빠르게 처리 할 때(시간 단축) 
            
# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
# print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

