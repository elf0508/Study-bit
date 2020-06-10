# RandomForest (랜덤포래스트)
# 분류 : iris
# 사이킷런 버전 : 22.1로 해야한다.

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  # 분류 : iris

#1. 데이터
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.2, 
#                                                     shuffle = True, random_state = 43)


# grid / random search에서 사용할 매개 변수
# 리스트 형대 - 키 : 벨류 딕셔너리
# parameters = [
#     {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear']},
#     {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['rbf'], 'svm__gamma':[0.001, 0.0001]},
#     {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear'], 'svm__gamma':[0.001, 0.0001]}
# ]
# Pipeline 엮게될때, 파라미터 앞에 : '이름(소문자)__' 명시해야 한다.
#                                       모델명__파라미터

# parameters = [
#     {'C':[1, 10, 100, 1000], 'kernel':['linear']},
#     {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
#     {'C':[1, 10, 100, 1000], 'kernel':['linear'], 'gamma':[0.001, 0.0001]}
# ]

parameters = [
    {'randomforestclassifier__n_jobs':[1], 'randomforestclassifier__max_depth':[11,12,13]},
    {'randomforestclassifier__min_samples_leaf':[1, 10, 100], 'randomforestclassifier__min_samples_split':[10,11,12]},
    {'randomforestclassifier__n_jobs':[1],'randomforestclassifier__max_depth':[13,14,15]}
]

#2. 모델
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])    
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svc', SVC())])    
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
#                      전처리,            모델 명시

model = RandomizedSearchCV(pipe, parameters , cv = 5)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가,예측 (evaluate, predict)
acc = model.score(x_test, y_test)
print("========================")
print('최적의 매개변수 = ', model.best_estimator_)
print("========================")
print('최적의 매개변수 = ', model.best_params_)
print("========================")

print('acc : ', acc)

import sklearn as sk
print("sklearn : ", sk.__version__)  # sklearn :  0.22.1

'''
최적의 매개변수 =  Pipeline(memory=None,
         steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                ('svm',
                 SVC(C=1000, break_ties=False, cache_size=200,
                     class_weight=None, coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma=0.001,
                     kernel='rbf', max_iter=-1, probability=False,
                     random_state=None, shrinking=True, tol=0.001,
                     verbose=False))],
         verbose=False)
acc :  1.0
'''