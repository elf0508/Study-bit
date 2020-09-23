# keras44_save.py / 모델 저장 : h5확장자를 사용한다.
# 40번을 카피해서 복붙


import numpy as np
from keras.models import Sequential #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, LSTM # Dense와 LSTM 레이어를 쓰겠다


# 2. 모델구성 

model = Sequential()

model.add(LSTM(10, input_shape=(4, 1)))

model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(16)) 
 
model.add(Dense(10))

model.summary()

model.save(".//model//save_keras44.h5")
# model.save("./model/save_keras44.h5")
# model.save(".\model\save_keras44.h5")

print("저장 잘됬다.")