# 실무에서 데이터 분류 축소

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기

wine = pd.read_csv('./data/csv/winequality-white.csv',
                   header = 0, index_col = None,
                   sep = ';', encoding = 'cp949')


y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.shape)  # (4898, 11)
print(y.shape)  # (4898,)


# y레이블 축소

newlist = []
for i in list(y):  # quality => 많았던 등급을 3개 등급(0,1,2)으로 축소 시켰다.
    if i <= 4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else: 
        newlist +=[2]

y = newlist

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 2.모델

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# 3.훈련
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

# 4.평가,예측

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_pred))
print("acc : ", acc)

# acc_score :  0.939795918367347
# acc :  0.939795918367347
