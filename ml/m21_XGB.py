# XGBClassifier

#유방암 - 컬럼 30개
# 이진분류

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


cancer = load_breast_cancer()
x_train, x_test, y_train,y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 42
)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier()


# max_features : 기본값 써라!
# n_estimators : 클수록 좋다! 단점: 메모리 짱 차지, 기본값 100
# n_jobs = -1 : 병렬처리

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

'''
[0.00807526 0.02399339 0.01387195 0.01999373 0.00639482 0.0053644
 0.02831325 0.400554   0.00210795 0.00413082 0.01327085 0.00458036
 0.01822196 0.00435152 0.00383945 0.00356999 0.02908346 0.00209119
 0.0014143  0.00234754 0.054863   0.01614437 0.05690607 0.06654113
 0.00544669 0.00221457 0.01547522 0.1840091 
 0.00282953 0.        ]
 
acc :  0.956140350877193
'''

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
              align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()




