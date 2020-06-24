
# 회귀모델이다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from xgboost import XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.interpolate import interp1d
from keras.layers import LeakyReLU
import pickle 

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

leaky = LeakyReLU(alpha = 0.2)

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
scaler = RobustScaler()
# scaler = MinMaxScaler()
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
# print(x_train.shape)          
# print(x_test.shape)          


## 원핫인코딩 : 다중분류에서만 사용/'순서'가 아니라, '분류'일뿐이다라고 컴에게 알려줌
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)               
# print(y_test.shape)   

# 2. 모델

from keras.models import Sequential, Input
from keras.layers import Dense, Dropout

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

model =  XGBRegressor(n_estimators=1000, learning_rate=0.1)
              
model.fit(x_train, y_train, verbose=True, eval_metric='error',
                eval_set=[(x_train, y_train), (x_test, y_test)])


result = model.evals_result()
# print("eval's results :", result)

y_pred = model.predict(x_test)

# print(multi_XGB.estimators_[0].feature_importances_)
# print(multi_XGB.estimators_[1].feature_importances_)
# print(multi_XGB.estimators_[2].feature_importances_)
# print(multi_XGB.estimators_[3].feature_importances_)

    
from pandas import Series, DataFrame
import numpy as np
import pandas as pd

# pandas
def outliers(data_out):
        quartile_1 = data_out.quantile(.25)
        quartile_3 = data_out.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out > upper_bound) | (data_out < lower_bound))
         
a3 = pd.DataFrame({'a' : [1, 3, 5, 200, 100, 8],
                    'b' : [300, 100, 6, 8, 2, 3]})
print(77.0 * 1.5)

b3 = outliers(a3)
print(b3)

multi_XGB = XGBRegressor(multi_XGB.estimators_)

for i in range(len(multi_XGB.estimators_)):
    threshold = np.sort(multi_XGB.estimators_[i].feature_importances_)[18:]
    print(threshold)
    

    for thres in threshold:
        selection = SelectFromModel(multi_XGB.estimators_[i], threshold = thres, prefit = True)
        
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)
        print(select_x_train.shape[1])
        
        parameter = {
            'n_estimators': [100, 200, 400],
            'learning_rate' : [0.05, 0.07, 0.1],
            'colsample_bytree': [ 0.7, 0.8, 0.9],
            'colsample_bylevel':[ 0.7, 0.8, 0.9],
            'max_depth': [4, 5, 6]
        }
        # search = RandomizedSearchCV( XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist'), parameter, cv =5)
        search = RandomizedSearchCV( XGBRegressor(), parameter, cv =5)
        multi_search = MultiOutputRegressor(search,n_jobs = -1)
        
        multi_search.fit(select_x_train, y_train )
        
        y_pred = multi_search.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)
        score =r2_score(y_test, y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thres, select_x_train.shape[1], score*100.0, mae))
        
        y_predict = multi_search.predict(select_x_pred)

        # submission
        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./dacon/select_XGB03_%i_%.5f.csv'%(i, mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')



###############

# y_predict = pd.DataFrame({
#   'id' : np.array(range(10000, 20000)),
#   'hhb': y_predict[:,0],
#   'hbo2': y_predict[:, 1],
#   'ca': y_predict[:, 2],
#   'na':y_predict[:, 3]
# })
# y_predict.to_csv('./dacon/sample_submission.csv', index = False )


# 서브밋파일 만든다.
# .to_csv(경로)
# 제출




