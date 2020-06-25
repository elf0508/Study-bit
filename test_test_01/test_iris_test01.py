import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터
iris = datasets.load_iris()

# 데이터 전처리 : 지금은 그냥 머신러닝에 바로 적용했다. 

# 3, 4번째 특징 추출
x = iris.data[:, [2,3]]

# 클래스 라벨을 가져온다.
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(

    x, y, test_size = 0.3, random_state = 0
)

# x_train, x_test, y_train, y_test = train_test_split(

#     x, y, test_size = 0.2, random_state = 0
# )

# 모델

svc = svm.SVC(C = 1, kernel = 'rbf', gamma = 0.001)

# 훈련

svc.fit(x_train, y_train)

# 평가

y_pred = svc.predict(x_test)

print("acc : %.2f" % accuracy_score(y_test, y_pred))

# 0.3 --> acc : 0.60
# 0.2 --> acc : 0.57