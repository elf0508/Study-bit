from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

x, y = load_breast_cancer(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = XGBClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['logloss', 'error'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

# Thresh=0.001, n = 30, ACC : 96.49%
# Thresh=0.002, n = 29, ACC : 96.49%
# Thresh=0.002, n = 28, ACC : 96.49%
# Thresh=0.003, n = 27, ACC : 96.49%
# Thresh=0.004, n = 26, ACC : 96.49%
# Thresh=0.004, n = 25, ACC : 96.49%
# Thresh=0.004, n = 24, ACC : 96.49%
# Thresh=0.004, n = 23, ACC : 96.49%
# Thresh=0.005, n = 22, ACC : 96.49%
# Thresh=0.005, n = 21, ACC : 96.49%
# Thresh=0.006, n = 20, ACC : 96.49%
# Thresh=0.006, n = 19, ACC : 96.49%
# Thresh=0.008, n = 18, ACC : 96.49%
# Thresh=0.008, n = 17, ACC : 96.49%
# Thresh=0.008, n = 16, ACC : 96.49%
# Thresh=0.010, n = 15, ACC : 96.49%
# Thresh=0.011, n = 14, ACC : 96.49%
# Thresh=0.014, n = 13, ACC : 96.49%
# Thresh=0.015, n = 12, ACC : 96.49%
# Thresh=0.017, n = 11, ACC : 96.49%
# Thresh=0.018, n = 10, ACC : 96.49%
# Thresh=0.019, n = 9, ACC : 96.49%
# Thresh=0.020, n = 8, ACC : 96.49%
# Thresh=0.026, n = 7, ACC : 96.49%
# Thresh=0.032, n = 6, ACC : 97.37%
# Thresh=0.082, n = 5, ACC : 95.61%
# Thresh=0.110, n = 4, ACC : 94.74%
# Thresh=0.123, n = 3, ACC : 96.49%
# Thresh=0.166, n = 2, ACC : 92.11%
# Thresh=0.267, n = 1, ACC : 88.60%

'''
from xgboost import XGBRegressor, XGBClassifier, plot_importance # 중요한걸 그리겠네
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np

# 이중분류 모델
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

# XGB에만 있는 애들인거 같아
model = XGBClassifier(n_estimators=300, learning_rate=0.01) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['error','auc'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=50) 
result = model.evals_result_
 

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :%.2f%%' %(acc*100.0))
print('acc:', acc)

thresholds = np.sort(model.feature_importances_)

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