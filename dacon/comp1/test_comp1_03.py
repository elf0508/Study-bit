import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error as MAE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor

test = pd.read_csv('./dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
print(x_train.shape)
print(y_train.shape)
# print(x_test.shape)

# 2. model
final_y_test_pred = []
final_y_pred = []
parameter = [
    {'n_estimators': [1000],
    'learning_rate': [0.045,0.055,0.065,0.075,0.085],
    'max_depth': [5,6,7]},
    {'n_estimators': [1000],
    'learning_rate': [0.045,0.055,0.065,0.075,0.085],
    'colsample_bytree':[0.6,0.65,0.7,0.75,0.8,0.85],
    'max_depth': [5,6,7]},
    {'n_estimators': [1000],
    'learning_rate': [0.045,0.055,0.065,0.075,0.085],
    'colsample_bylevel': [0.6,0.65,0.7,0.75,0.8,0.85],
    'max_depth': [5,6,7]}
]
# parameter =[
#     {'n_estimators': [500],
#     'learning_rate': [0.075],
#     'colsample_bylevel': [0.75],
#     'max_depth': [6]},
#     {'n_estimators': [500],
#     'learning_rate': [0.075],
#     'colsample_bytree': [0.75],
#     'max_depth': [6]},
#     {'n_estimators': [500],
#     'learning_rate': [0.075],
#     'max_depth': [6]}
# ]

settings = {
    'verbose': False,
    'eval_metric': ['logloss','mae'],
    'eval_set' : [(x_train, y_train), (x_test,y_test)],
    # 'early_stopping_rounds' : 30
}

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

# 모델 컬럼별 4번
for i in range(4):
    model = XGBRegressor()
    settings['eval_set'] = [(x_train, y_train), (x_test,y_test)]
    model.fit(x_train,y_train[:,i], **settings)
    y_test_pred = model.predict(x_test)
    score = model.score(x_test,y_test[:,i])
    mae = MAE(y_test[:,i], y_test_pred)
    print("r2 : ", score)
    print("mae :", mae)
    thresholds = np.sort(model.feature_importances_)[[i for i in range(0,176,35)]]
    print(thresholds)
    best_mae = mae
    best_model = model
    best_y_pred = model.predict(x_pred)
    best_y_test_pred = y_test_pred
    print(best_y_pred.shape)
    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 이 둘중 하나 쓰는거 이해하면 사용 가능
                                                ## 이거 주어준 값 이하의 중요도를 가진 feature를 전부 자르는 파라미터
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)

        print(select_x_train.shape)

        selection_model = RandomizedSearchCV(XGBRegressor(), parameter, n_jobs=-1, cv = kfold,n_iter=10)
        settings['eval_set'] = [(select_x_train, y_train), (select_x_test,y_test)]
        selection_model.fit(select_x_train, y_train[:,i], **settings)

        y_pred = selection_model.predict(select_x_test)
        r2 = r2_score(y_test[:,i],y_pred)
        mae = MAE(y_test[:,i],y_pred)
        print(selection_model.best_params_)
        if mae <= best_mae:
            print("예아~")
            best_mae = mae
            best_model = selection_model
            best_y_pred = selection_model.predict(select_x_pred)
            best_y_test_pred = y_pred
        print("Thresh=%.3f, n=%d, MAE: %.5f" %(thresh, select_x_train.shape[1], mae))
    final_y_pred.append(best_y_pred)
    final_y_test_pred.append(best_y_test_pred)

print('MAE :', MAE(y_test, np.array(final_y_test_pred).T))

final_y_pred = np.array(final_y_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": final_y_pred[0,:],
    "hbo2": final_y_pred[1,:],
    "ca": final_y_pred[2,:],
    "na": final_y_pred[3,:]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)