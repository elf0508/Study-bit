# kfold : 교차 검증 ( 신뢰도는 높은편이다.)
# 모든 데이터들을 조각 조각 내서 train & test 로 갈라주는 것
# kfold  : train_test_split 의 업그레이드 버전 + validation

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

# 1.데이터

iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]  
y = iris.iloc[:, 4]

print(x)
print(y)

warnings.filterwarnings('ignore')

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
#                                                         random_state=44)

warnings.filterwarnings('ignore')

kfold = KFold(n_splits=5, shuffle=True)

allAlorithms = all_estimators(type_filter='classifier') # classifier : 분류

for (name, algorithm) in allAlorithms:
    model = algorithm()

    scores = cross_val_score(model, x, y, cv=kfold)

    print(name, "의 정답률은 = ")
    print(scores)  # scores = acc

import sklearn
print(sklearn.__version__)
