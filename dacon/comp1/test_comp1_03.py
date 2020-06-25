
# 회귀모델이다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

x_train = pd.read_csv('./dacon/comp1/train.csv', header=0, index_col=0)
y_train = pd.read_csv('./dacon/comp1/test.csv', header=0, index_col=0)
x_pred = pd.read_csv('./dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('x_train.shape : ', x_train.shape)  # (10000, 75) : x_train, x_test
print('y_train.shape : ', y_train.shape)   # (10000, 71) : x_predict
print('x_pred.shape : ', x_pred.shape)  # (10000, 4) : y_predict

# 결측치 제거
print(x_train.isnull().sum())

x_train = x_train.interpolate()  # 보간법- 선형보간
print(x_train.isnull().sum())

y_train = y_train.interpolate()  
print(y_train.isnull().sum())

##########################################

x_train = x_train.fillna(method='bfill') 
y_train = y_train.fillna(method='bfill')

print(y_train.isnull().sum())
print("=" * 40)
print(x_train.isnull().sum())
print(y_train.isnull().sum())

x_train = x_train.iloc[:, :71]
y_train = y_train.iloc[:, 71:]

print(x_train.shape)          
print(y_train.shape)         


np.save('./dacon/x_data', arr = x_train)
np.save('./dacon/y_data', arr = y_train)
np.save('./dacon/y_data', arr = x_pred )


# numpy 데이터 로드
x_train = np.load('./dacon/x_data.npy')
y_train = np.load('./dacon/y_data.npy')
x_pred  = np.load('./dacon/y_data.npy')

print(x_train.shape)          
print(y_train.shape)         
print(x_pred .shape)          


# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, random_state = 1,
    train_size = 0.75, shuffle = True)
print(x_train.shape)        # (8000, 71)
print(x_test.shape)         # (2000, 71)
print(y_train.shape)        # (8000, 4)
print(y_test.shape)         # (2000, 4)


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])
print(x_test[0])


# 2. 모델

parameters =[
    {'n_estimators': [1000],
    'learning_rate': [0.025],
    'colsample_bylevel': [0.75],
    'eval_metric': ['mae'],
    'max_depth': [6]
    }
]
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
search = RandomizedSearchCV(XGBRegressor(), parameters, cv = kfold, n_iter=1, n_jobs=-1)
search = MultiOutputRegressor(search)

search.fit(x_train, y_train)

print(search.estimators_)
y_test_pred = search.predict(x_test)

# print(search.best_params_)
print("R2 :", r2_score(y_test,y_test_pred))
print("mae :", mae(y_test,y_test_pred))

############  위에 부분까지는 나옴 ##############

x_train = x_train.values
x_test= x_test.values
y_train = y_train.values

y_pred = search.predict(x_pred)

submissions = pd.DataFrame({
    'id' : np.array(range(10000, 20000)),
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)


# 서브밋파일 만든다.
# .to_csv(경로)
# 제출




