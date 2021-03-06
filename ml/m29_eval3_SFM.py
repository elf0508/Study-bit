from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time
x, y = load_iris(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

# model = XGBClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1)
model = XGBClassifier(gpu_id=0, tree_method='gpu_hist')

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

start = time.time()

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    # selection_model = XGBClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 
    selection_model = XGBClassifier(gpu_id=0, tree_method='gpu_hist')

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['mlogloss', 'merror'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

end = time.time() - start
print(" 걸린 시간 :", end)

# model = XGBClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1)
#           의 결과 값

# Thresh=0.022, n = 4, ACC : 100.00%
# Thresh=0.041, n = 3, ACC : 100.00%
# Thresh=0.463, n = 2, ACC : 100.00%
# Thresh=0.473, n = 1, ACC : 93.33%
#  걸린 시간 : 0.34758806228637695

# acc :100.00%

# model = XGBClassifier(gpu_id=0, tree_method='gpu_hist')의 결과 값

# Thresh=0.023, n = 4, ACC : 100.00%
# Thresh=0.024, n = 3, ACC : 100.00%
# Thresh=0.442, n = 2, ACC : 100.00%
# Thresh=0.512, n = 1, ACC : 93.33%
#  걸린 시간 : 0.8088362216949463


'''

from xgboost import XGBRegressor, XGBClassifier, plot_importance 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np

# 다중분류 모델
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

model = XGBClassifier(objective='multi:softmax', estimators=300, learning_rate=0.01, n_jobs=-1) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['merror'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=20) 

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