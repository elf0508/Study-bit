'''
m28_eval
1. eval 에  'loss'와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. plot으로 그릴 것

m29_eval
SelectFromModel에 
1. 회귀
2. 이진 분류
3. 다중 분류

1. eval 에  'loss'와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. plot으로 그릴 것

4. 결과는 주석으로 소스 하단에 표시.

5. m27 ~ 29까지 완벽 이해할 것!
'''
from xgboost import XGBRegressor, plot_importance  
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# 회귀 모델
x, y = load_boston(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                            shuffle = True, random_state = 66)

# model = XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 
model = XGBRegressor(gpu_id=0, tree_method='gpu_hist',  n_jobs = -1)
model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    # selection_model = XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 
    selection_model = XGBRegressor(gpu_id=0, tree_method='gpu_hist',  n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['logloss', 'rmse'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    r2 = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thres, select_x_train.shape[1], r2*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

# model = XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 의 결과 값

# Thresh=0.003, n = 13, R2 : 93.54%
# Thresh=0.005, n = 12, R2 : 93.71%
# Thresh=0.006, n = 11, R2 : 93.69%
# Thresh=0.009, n = 10, R2 : 93.78%
# Thresh=0.012, n = 9, R2 : 94.11%
# Thresh=0.014, n = 8, R2 : 94.31%
# Thresh=0.015, n = 7, R2 : 93.76%
# Thresh=0.017, n = 6, R2 : 92.80%
# Thresh=0.017, n = 5, R2 : 93.63%
# Thresh=0.039, n = 4, R2 : 92.26%
# Thresh=0.045, n = 3, R2 : 89.30%
# Thresh=0.248, n = 2, R2 : 81.05%
# Thresh=0.569, n = 1, R2 : 69.21%

# model = XGBRegressor(gpu_id=0, tree_method='gpu_hist',  n_jobs = -1) 의 결과 값

# Thresh=0.003, n = 13, R2 : 91.54%
# Thresh=0.004, n = 12, R2 : 91.14%
# Thresh=0.008, n = 11, R2 : 90.37%
# Thresh=0.009, n = 10, R2 : 91.43%
# Thresh=0.016, n = 9, R2 : 91.75%
# Thresh=0.016, n = 8, R2 : 91.94%
# Thresh=0.020, n = 7, R2 : 91.21%
# Thresh=0.023, n = 6, R2 : 90.99%
# Thresh=0.039, n = 5, R2 : 90.61%
# Thresh=0.057, n = 4, R2 : 90.58%
# Thresh=0.066, n = 3, R2 : 91.39%
# Thresh=0.309, n = 2, R2 : 84.13%
# Thresh=0.429, n = 1, R2 : 68.82%



'''
m28_eval1   _boston_회귀
m28_eval2   _cancer_이진분류
m28_eval3   _iris_다중분류
만들것

SelectFromModel 적용시켜서
1. 회귀     m29_eval1   _boston_회귀
2. 이진분류 m29_eval2   _cancer_이진분류
3. 다중분류 m29_eval3   _iris_다중분류

1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlyStopping 적용
# 3. plot으로 그릴 것

4. 결과는 주석으로 소스 하단에 표시

m27 ~ 29까지 완벽 이해할 것


# 이진분류_boston
from xgboost import XGBRegressor, plot_importance # 중요한걸 그리겠네
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
# 이중분류 모델
dataset = load_boston()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)


model = XGBRegressor(n_estimators=100, learning_rate=0.01) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['logloss','rmse'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=100) 

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :%.2f%%' %(r2*100.0))

thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds : # 컬럼수만큼 돈다! 빙글빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape) # 칼럼이 하나씩 줄고 있는걸 알 수 있음 (가장 중요 x를 하나씩 지우고 있음)
    
    selection_model = XGBRegressor()
    selection_model.fit(selection_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print('R2는',score)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))
'''