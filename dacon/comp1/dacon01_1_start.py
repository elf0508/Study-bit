
# 회귀모델이다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

train = pd.read_csv('./dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape : ', train.shape)  # (10000, 75) : x_train, x_test
print('test.shape : ', test.shape)   # (10000, 71) : x_predict
print('submission.shape : ', submission.shape)  # (10000, 4) : y_predict

# 결측치 제거
print(train.isnull().sum())

train = train.interpolate()  # 보간법- 선형보간
print(train.isnull().sum())
test = test.interpolate()  
print(test.isnull().sum())

##########################################

train = train.fillna(method='bfill') 
test = test.fillna(method='bfill') 
print(test.isnull().sum())
print("=" * 40)
print(train.isnull().sum())
print(test.isnull().sum())

x = train.iloc[:, :71]
y = train.iloc[:, 71:]
print(x.shape)          # (10000, 71)
print(y.shape)          # (10000, 4)


# np.save('./dacon/x_data', arr = x)
# np.save('./dacon/y_data', arr = y)


# # numpy 데이터 로드
# x = np.load('./dacon/x_data.npy')
# y = np.load('./dacon/y_data.npy')
# print(x.shape)          # (10000, 71)
# print(y.shape)          # (10000, 4)


# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 1,
    train_size = 0.75, shuffle = True)
print(x_train.shape)        # (8000, 71)
print(x_test.shape)         # (2000, 71)
print(y_train.shape)        # (8000, 4)
print(y_test.shape)         # (2000, 4)


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])
print(x_test[0])


# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)
# print(x_train.shape)   # (7500, 2)
# print(x_test.shape)    # (2500, 2)

import seaborn as sns 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb 
from sklearn.model_selection import KFold 

model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBRegressor()


plt.figure(figsize=(4, 12))
sns.heatmap(train.corr().loc['rho':'990_dst', 'hhb':].abs())
# plt.show()

x_train = train.loc[:, '650_dst':'990_dst']
y_train = train.loc[:, 'hhb':'na']

print(x_train.shape) # (10000, 35)
print(y_train.shape) # (10000, 4)

# 모델 학습 및 검증
# XGBoost를 이용하여 모델 생성
# 학습 데이터셋을 5개로 나눠 교차검증

def train_model(x_data, y_data, k=5):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data.iloc[train_idx], y_data[train_idx]
        x_val, y_val = x_data.iloc[val_idx], y_data[val_idx]
    
        d_train = xgb.DMatrix(data = x_train, label = y_train)
        d_val = xgb.DMatrix(data = x_val, label = y_val)
        
        wlist = [(d_train, 'train'), (d_val, 'eval')]
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed':777
            }

        model = xgb.train(params=params, dtrain=d_train, num_boost_round=500, verbose_eval=500, evals=wlist)
        models.append(model)
    
    return models

models = {}
for label in y_train.columns:
    print('train column : ', label)
    models[label] = train_model(x_train, y_train[label])

for col in models:
    preds = []
    for model in models[col]:
        preds.append(model.predict(xgb.DMatrix(test.loc[:, '650_dst':])))
    pred = np.mean(preds, axis=0)

    submission[col] = pred

# submission.head()

# submission.to_csv('Dacon_baseline.csv', index=False)

# model.fit(x_train, y_train)

mae = model.score(x_test, y_test)

print(model.feature_importances_)
print("mae : ", pred)

# xgboost 나 GradientBoosting 할 때, shape 안맞다고 에러 뜰 때
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(xgb.XGBRFRegressor())
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)
y4 = model.predict(test.values)

# DecisionTreeRegressor acc :  -0.007782003951759658
# DecisionTreeRegressor : mae 는 1.7070600986480713
# RandomForestRegressor : mae 는 2.1446380615234375
# GradientBoostingRegressor : mae 는 1.7161730527877808
#  XGBRegressor : mae 는 1.6988283395767212


'''
## 원핫인코딩 : 다중분류에서만 사용/'순서'가 아니라, '분류'일뿐이다라고 컴에게 알려줌
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)               
# print(y_test.shape)   
'''
'''
# 2. 모델

from keras.models import Sequential, Input
from keras.layers import Dense, Dropout

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

model = Sequential()

model.add(Dense(128, input_shape= (71, ),
                activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(rate = 0.15))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))

model.summary()

warnings.filterwarnings('ignore')


# 3. 컴파일 및 훈련
model.compile(loss ='mae', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 100,
         batch_size = 20, validation_split = 0.2,
         callbacks = [es])  
         

#4. 평가와 예측
loss,mae = model.evaluate(x_test,y_test) 
print('mae 는', mae)
'''

'''
test = test.values  # 넘파이 형식으로 변환
y_predict = model.predict(test)
print(y_predict)
'''

'''
y_predict = pd.DataFrame(y_predict) # 판다스로 변환해서,csv로 저장
print(type(y_predict))
# print(x_predict)
'''

# y_predict = pd.DataFrame({
#   'id' : np.array(range(10000, 20000)),
#   'hhb': y_predict[:,0],
#   'hbo2': y_predict[:, 1],
#   'ca': y_predict[:, 2],
#   'na':y_predict[:, 3]
# })
# y_predict.to_csv('./dacon/sample_submission_1.csv', index = False )


# 서브밋파일 만든다.
# .to_csv(경로)
# 제출



