import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('kaggle/titanic/train.csv')


print(df.shape)
print(df.head())

df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Cabin'].fillna('n', inplace=True)

df['Embarked'].fillna('S', inplace=True)

df['Cabin'] = df['Cabin'].str[:1]

df['Cabin'].head()

df[df['Age'] <= 0]

def age(x):

    Age = ''
    if x <= 12: Age='Child'
    elif x <= 19: Age='Teen'
    elif x <= 30: Age='Young_adult'
    elif x <= 60: Age='Adult'
    else: Age='Old'
        
    return Age


df.Age.isnull().any()

# encoding

from sklearn.preprocessing import LabelEncoder

def encoding(x):

    for i in ['Sex', 'Age', 'Cabin', 'Embarked']:

        x[i] = LabelEncoder().fit_transform(x[i])
        
    return x   

df = encoding(df)

df.head()


df = pd.get_dummies(df, columns=['Pclass', 'Sex', 'Age', 'Embarked'])

df.head()

y = df['Survived']

X = df.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# 교차검증

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

xgb = XGBClassifier(random_state=44)
gb = GradientBoostingClassifier(random_state=44)
rf = RandomForestClassifier(random_state=44)

xgb_cross = cross_val_score(xgb, X, y, cv=5, verbose=1)
gb_cross = cross_val_score(gb, X, y, cv=5, verbose=1)
rf_cross = cross_val_score(rf, X, y, cv=5, verbose=1)

for count, accuracy in enumerate(xgb_cross):

    print('XGB {}번째 accuracy : {:.3f}'.format(count, accuracy))

print('XGB 평균 성능 : {:.3f}'.format(np.mean(xgb_cross)))

print('--------------------------------------')

for count, accuracy in enumerate(gb_cross):

    print('GB {}번째 accuracy : {:.3f}'.format(count, accuracy))

print('GB 평균 성능 : {:.3f}'.format(np.mean(gb_cross)))

print('--------------------------------------')

for count, accuracy in enumerate(rf_cross):

    print('RF {}번째 accuracy : {:.3f}'.format(count, accuracy))

print('RF 평균 성능 : {:.3f}'.format(np.mean(rf_cross)))



# 그리드서치

from sklearn.model_selection import GridSearchCV

xgb_param = {

    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

gb_param = {

#     'n_estimators': [100],
    'min_samples_leaf': [7, 9, 13],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.05, 0.02, 0.01],
}

grid_xgb = GridSearchCV(xgb, param_grid=xgb_param, scoring='accuracy', cv=5)

grid_gb = GridSearchCV(gb, param_grid=gb_param, scoring='accuracy', cv=5)

grid_xgb.fit(X_train, y_train)

grid_gb.fit(X_train, y_train)

print('xgboost best param : ',grid_xgb.best_params_)
print('xgboost best accuracy : ',grid_xgb.best_score_)
print('gradient boosting best param : ',grid_gb.best_params_)
print('gradient boosting best accuracy : ',grid_gb.best_score_)

# best parameter로 학습된 모델로 테스트 데이터 예측 및 평가

xgb_pred = grid_xgb.best_estimator_.predict(X_test)
gb_pred = grid_gb.best_estimator_.predict(X_test)

print('xgboost accuracy(test set) : {:.3f}'.format(accuracy_score(y_test, xgb_pred)))
print('gradient boosting accuracy(test set) : {:.3f}'.format(accuracy_score(y_test, gb_pred)))

# 랜덤포레스트

rf.fit(X_train, y_train)

print('randomforest accuracy(test set) : {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))


submission = pd.DataFrame({

        "PassengerId": df["PassengerId"],
        "Survived":  rf.predict
    })

submission.to_csv('kaggle/titanic/titanic_05.csv', index=False)