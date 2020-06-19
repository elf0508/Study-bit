# GradientBoosting
#유방암 - 컬럼 30개
# 이진분류

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



cancer = load_breast_cancer()
x_train, x_test, y_train,y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 42
)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = GradientBoostingClassifier()
# model = GradientBoostingClassifier(max_depth=4)

# max_features : 기본값 써라!
# n_estimators : 클수록 좋다! 단점: 메모리 짱 차지, 기본값 100
# n_jobs = -1 : 병렬처리

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

'''
[0.0230886  0.01486293 0.04200334 0.04152124 0.00714661 0.01713181
 0.03332033 0.14498254 0.00310154 0.00455974 0.01364697 0.00461881
 0.00688051 0.04321067 0.0029681  0.00440204 0.0050146  0.00550111
 0.00630548 0.00742581 0.1168985  0.02601776 0.11388116 0.13176124
 0.01610832 0.00748019 0.02198572 0.11618584 0.01087693 0.00711158]

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




