import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

lr = LeakyReLU(alpha = 0.2)

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

x = pd.read_csv('./dacon/comp3/train_features.csv', header=0, index_col=0)                                                     
y = pd.read_csv('./dacon/comp3/train_target.csv', header=0, index_col=0)                                                       
x_pred = pd.read_csv('./data/dacon/comp3/test_features.csv', header=0, index_col=1)
                                                       
submission = pd.read_csv('./data/dacon/comp3/sample_submission.csv', header=0, index_col=0)

print("==== train_features ====")                                                                   
print(x)
print(x.shape)  # (1050000, 5)

print("==== train_target ====")
print(y)
print(y.shape)  # (2800, 4)

print("==== test_features ====")
print(x_pred)
print(x_pred.shape)  # (262500, 5)

#결측값은 없다

print(x.isnull().sum()) #없음
print(y.isnull().sum()) #없음
print(x_pred.isnull().sum()) #없음

# 판다스 --> 넘파이 파일로 바꿔주는 것 : .values
x = x.values
y = y.values
x_pred = x_pred.values

#저장하기

np.save('./data/dacon/comp3/train_features.npy', arr = x)
np.save('./data/dacon/comp3/train_target.npy', arr = y)
np.save('./data/dacon/comp3/test_features.npy', arr = x_pred)

#불러오기
x = np.load('./data/dacon/comp3/train_features.npy')
y = np.load('./data/dacon/comp3/train_target.npy')
x_pred = np.load('./data/dacon/comp3/test_features.npy')

print(type(x))          # <class 'numpy.ndarray'>
print(type(y))
print(type(x_pred))


# 스칼라
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()

scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(2800, -1, 5)

x_pred = x_pred.reshape(-1, 375, 5)

print(x.shape)    # (2800, 375, 4)   
print(y.shape)    # (2800, 4)
print(x_pred.shape) # (700, 375, 5)


'''

# 데이터 나누기

x_train, x_test, y_train, y_test = train_test_split(

    x, y, random_state = 1,

    train_size = 0.8, shuffle = True) 

print(x_train.shape)

'''

'''

# PCA 나누기

pca = PCA(n_components=2)

pca.fit(x_train)

x_train = pca.transform(x_train)

x_test = pca.transform(x_test)

print(x_train.shape)  #  (787500, 5)     

print(x_test.shape)    # (262500, 5) 

'''

# 2.모델

from keras.layers import LeakyReLU

leaky = LeakyReLU(alpha = 0.2)

model = Sequential()

model.add(LSTM(128, input_shape= (375, 5),
               activation = lr))
model.add(Dropout(rate = 0.5))

model.add(Dense(100, activation = lr))
model.add(Dropout(rate = 0.2))

model.add(Dense(64, activation = lr))
model.add(Dense(34, activation = lr))
model.add(Dropout(rate = 0.15))

model.add(Dense(32, activation = lr))
model.add(Dense(22, activation = lr))
model.add(Dropout(rate = 0.1))

model.add(Dense(16, activation = lr))
model.add(Dropout(rate = 0.1))

model.add(Dense(4, activation = lr))

model.summary()

warnings.filterwarnings('ignore')


# 3. 컴파일 및 훈련

model.compile(loss ='mae', optimizer = 'adam', metrics = ['mae'])

model.fit(x, y, epochs = 1,
         batch_size = 32, validation_split = 0.2,
         callbacks = [es])  



#4. 평가와 예측

# loss,mae = model.evaluate(x_test,y_test) 

# print('mae 는', mae)



y_predict = model.predict(x_pred)

print(y_predict)

# 판다스로 변환해서,csv로 저장

print(type(y_predict))

# print(x_predict)

submissions = pd.DataFrame({

  'id' : np.array(range(2800, 3500)),
  'X': y_predict[:,0],
  'Y': y_predict[:, 1],
  'M': y_predict[:, 2],
  'V':y_predict[:, 3]
})

submissions.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

# 서브밋파일 만든다.

# .to_csv(경로)

# 제출