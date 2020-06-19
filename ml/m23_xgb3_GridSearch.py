# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 다중분류 모델

dataset = load_iris()

x = dataset.data
y = dataset.target

print(x.shape) # (150, 4)
print(y.shape) # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                    shuffle = True, random_state = 66 )


# XGB 필수 파라미터 ##
n_estimators = 10000        #나무 100개/ decision tree보다 100배 느려
learning_rate = 0.001      # 학습률 디폴트 값/ 상당히 중요하다
colsample_bytree = 0.9
colsample_bylevel = 0.6   # 0.6~0.9
###
#점수 :  0.9

max_depth = 5
n_jobs = -1 #  딥러닝 빼고 default 로 써라


###############  여기서 부터

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
    "max_depth":[4, 5, 6]},

    {"n_estimators" : [90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
    "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]},

     {"n_estimators" : [90, 100, 110], "learning_rate":[0.1, 0.001, 0.5],
    "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}]


n_jobs = -1

# 트리- 전처리 안해, 결측치 제거 안해, 
# xgb - 속도 빨라, 일반 머신러닝보단 조금 느려/ 앙상블이라서
# 보간법 안해도 돼

# CV, feature importance 꼭 넣기

xgb = XGBClassifier()

model = GridSearchCV(XGBClassifier(), parameters, cv = 5, n_jobs= -1 )

model.fit(x_train, y_train)

#####################################  까지 복사

print("===================")
print(model.best_estimator_)
print("===================")
print(model.best_params_)

score = model.score(x_test, y_test)
print("점수 : ", score)
print("===================")
# print(model.feature_importances_)



# model.fit(x_train, y_train)

# score = model.score(x_test, y_test) # evaluate
# print('점수 : ', score)
# print(model.feature_importances_)

# plot_importance(model)
# # plt.show()

# f0~ f12 까지,  f12가 제일 중요

# 점수 :  -0.06904014604139475
# [0.03427136 0.00086752 0.01226326 0.         0.06205949 0.35779986
#  0.00939034 0.05117243 0.00469103 0.01540447 0.06899065 0.01284792
#  0.37024173]
'''
===================
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=90, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
===================
{'colsample_bytree': 0.6, 'learning_rate': 0.001, 'max_depth': 4, 'n_estimators': 90}
점수 :  0.9666666666666667
===================

################################################

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)


parameters = {
            "n_estimators":[90, 110, 200, 400], 
            "learning_rate": [0.1, 0.001,  0.5, 0.07, 0.05],
            "max_depth": [4, 5, 6, 7, 8],
            "colsample_bytree":[0.6, 0.9, 0.7, 1],
            "colsample_bylevel": [0.6, 0.7, 0.8, 0.9]
}

model = GridSearchCV(XGBClassifier(), parameters, cv =5, n_jobs = -1)

model.fit(x_train, y_train)

print("======================")
print(model.best_estimator_)
print('======================')
print(model.best_params_)
print('======================')

score = model.score(x_test, y_test)
print('점수 :', score)


======================
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=90, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
======================
{'colsample_bytree': 0.6, 'learning_rate': 0.001, 'max_depth': 4, 'n_estimators': 90} 
======================
점수 : 0.9666666666666667

'''