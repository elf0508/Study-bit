import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, KFold,  RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor

# es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

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

np.save('./dacon/x_data', arr = x)
np.save('./dacon/y_data', arr = y)

# numpy 데이터 로드

x = np.load('./dacon/x_data.npy')
y = np.load('./dacon/y_data.npy')

print(x.shape)          # (10000, 71)
print(y.shape)          # (10000, 4)


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

# 2. 모델

warnings.filterwarnings('ignore')

parameters =[

    {'n_estimators': [1000],

    'learning_rate': [0.075],

    'colsample_bylevel': [0.75],

    'max_depth': [6]}

]

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

search = RandomizedSearchCV(XGBRegressor( eval_metric='mae'), parameters, cv = kfold, n_iter=1, n_jobs=-1)

search = MultiOutputRegressor(search)

search.fit(x_train, y_train)

# print(search.best_params_)

print("R2 :", search.score(x_test,y_test))

y_pred = search.predict(submission)

submission = pd.DataFrame({

    'id' : np.array(range(10000, 20000)),
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]

})

# y_pred = search.predict(submission)

# submission = pd.DataFrame({
#     "id": test.index,
#     "hhb": y_pred[:,0],
#     "hbo2": y_pred[:,1],
#     "ca": y_pred[:,2],
#     "na": y_pred[:,3]
# })


submission.to_csv('./dacon/sample_submission1.csv', index = False)

# test = test.values  # 넘파이 형식으로 변환
# y_predict = model.predict(test)
# print(y_predict)

# y_predict = pd.DataFrame(y_predict) # 판다스로 변환해서,csv로 저장
# print(type(y_predict))
# print(x_predict)

# y_predict = pd.DataFrame({
#   'id' : np.array(range(10000, 20000)),
#   'hhb': y_predict[:, 0],
#   'hbo2': y_predict[:, 1],
#   'ca': y_predict[:, 2],
#   'na':y_predict[:, 3]
# })

# y_predict.to_csv('./dacon/sample_submission1.csv', index = False )

# a = np.arange(10000,20000)
# #np.arange--수열 만들때
# submission = y_predict
# submission = pd.DataFrame(submission, a)

# submission.to_csv("./dacon/sample_submission.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )

# 서브밋파일 만든다.
# .to_csv(경로)
# 제출