# pipeline : 데이터 처리 컴포넌트들이 연속되어 있는 것

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1.데이터
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                shuffle=True, random_state=43)


# 2.모델 
model = SVC()
# svc_model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 전처리와 모델을 같이 돌리는 것
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
#                          전처리,                  모델 명시
pipe = make_pipeline(MinMaxScaler(), SVC())


pipe.fit(x_train, y_train)

print("acc : ", pipe.score(x_test, y_test))

# acc :  0.9666666666666667