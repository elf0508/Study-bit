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

model = XGBClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

start = time.time()

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = XGBClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['mlogloss', 'merror'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)
    
    model.save_model("./model/xgb_save/thresh=%.3f-acc=%.2f.model"%(thres, acc))

end = time.time() - start
print(" 걸린 시간 :", end)

'''
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8, shuffle = True, random_state=66
)

parameter = [
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'max_depth': [4,5,6]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bytree':[0.6,0.68,0.9,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bylevel': [0.6,0.68,0.9,1],
    'max_depth': [4,5,6]}
]
fit_params = {
    'verbose':False,
    'eval_metric': ["mlogloss","merror"],
    'eval_set' :[(x_train, y_train), (x_test,y_test)],
    'early_stopping_rounds': 20
}
model = XGBClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("acc : ", score)


thresholds = np.sort(model.feature_importances_)

print(thresholds)
print(x_train.shape)
print("========================")

best_x_train = x_train
best_x_train = x_test
best_score = score
best_model = model

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median 이 둘중 하나 쓰는거 이해하면 사용 가능
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    print(select_x_train.shape)

    selection_model = GridSearchCV(XGBClassifier(), parameter, n_jobs=-1, cv = 5)
    fit_params['eval_set'] = [(select_x_train, y_train), (select_x_test,y_test)]
    selection_model.fit(select_x_train, y_train, **fit_params)

    y_pred = selection_model.predict(select_x_test)
    score = accuracy_score(y_test,y_pred)
    if best_score <= score:
        best_x_train = select_x_train
        best_x_test = select_x_test
        best_score = score
        best_model = selection_model
    print("Thresh=%.3f, n=%d, ACC: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

import joblib
joblib.dump(best_model, './model/xgb_Save/sfm3-'+str(best_score)+'.dat')


model2 = joblib.load('./model/xgb_Save/sfm3-'+str(best_score)+'.dat')

y_pred = model2.predict(best_x_test)
acc = accuracy_score(y_test,y_pred)
print('acc :', acc)
'''
