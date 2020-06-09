# gridSearch : 하이퍼 파라미터 자동화 
# RandomForest 적용
# breast_cancer 적용 

# GridSearchCV를 통한 랜덤 포레스트의 하이퍼 파라미터 튜닝
# https://injo.tistory.com/30
#RandomForest 적용

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV #여기서 cv는 cross_validation
from sklearn.metrics import accuracy_score
import warnings 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

#1. 데이터
# cifar10은 numpy
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
print("x.shape:",x.shape) #(569, 30)
print("y.shape:",y.shape) #(569, )

kfold=KFold(n_splits=5,shuffle=True) 

x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=60, test_size=0.2)

#RandomForest에서 제공하는 parameter
parameters={
    'n_estimators':[10,100],
    'max_depth':[6,8,10,12],
    'min_samples_leaf':[10,20,30],
    'min_samples_split':[10,20,30],
    }


kfold=KFold(n_splits=5, shuffle=True)
model=GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, n_jobs=-1) 
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("y_pred:",y_pred)

print("최종 정답률:",accuracy_score(y_test,y_pred))
print("최적의 매개변수:",model.best_estimator_)

#n_jobs : 다중 CPU제어
#n_jobs=-1 --> 모든 CPU다 쓰겠다(우리 학원은 6)

"""
최종 정답률: 0.9649122807017544
최적의 매개변수: RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=8, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=10, min_samples_split=20,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
            
 """

 

