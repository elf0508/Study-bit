# 저장하는 것

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      
print(y.shape)     

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBClassifier(n_estimators=1000, learning_rate=0.1)
              
model.fit(x_train, y_train, verbose=True, eval_metric='error',
                eval_set=[(x_train, y_train), (x_test, y_test)])


result = model.evals_result()
# print("eval's results :", result)


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred) 
print('acc : ', acc)

#########################################################

''' joblib_save'''

# import pickle      # 파이썬에서 제공한다.                                          # write b
# pickle.dump(model, open("./model/xgb_save/cancer.pickle.dat", "wb")) # model을 cancer.pickle.dat에 넣겠다.

# from joblib import dump, load
import joblib
# dump(model, "cancer.joblib.dat")
joblib.dump(model, "./model/xgb_save/cancer.joblib.dat")


print("저장됬다.")

''' joblib_load '''

# model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", "rb"))
model2 = joblib.load("./model/xgb_save/cancer.joblib.dat")
print("불러왔다.")

y_pred = model2.predict(x_test)
acc = accuracy_score(y_test, y_pred) 
print('acc : ', acc)