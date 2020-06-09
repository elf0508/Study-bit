import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings                                

warnings.filterwarnings('ignore')  # warnings이라는 에러에 대해서 넘어가겠다.

boston = pd.read_csv('D:/Study/data/csv/boston_house_prices.csv', header = 1)

print(boston)

x = boston.iloc[:, 0:13]
y = boston.iloc[:, 13]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

allAlegorithms = all_estimators(type_filter = 'regressor')  
               # : sklearn의 모든 classifier의 모델이 저장되어 있음

for (name, algorithm) in allAlegorithms:
    model = algorithm()

    model.fit(x_train, y_train)      # 훈련
    y_pred = model.predict(x_test)   # 예측
    print(name, "의 정답률 = ", r2_score(y_test, y_pred)) # acc


import sklearn
print(sklearn.__version__)

'''
Name: NOX, Length: 506, dtype: float64
ARDRegression 의 정답률 =  0.585925592719961
AdaBoostRegressor 의 정답률 =  0.743577255958432
BaggingRegressor 의 정답률 =  0.7923554577148922
BayesianRidge 의 정답률 =  0.6337185203038862
CCA 의 정답률 =  0.4422118554845459
DecisionTreeRegressor 의 정답률 =  0.7467068702435392
ElasticNet 의 정답률 =  0.24021389272672777
ElasticNetCV 의 정답률 =  0.6325286650228613
ExtraTreeRegressor 의 정답률 =  0.6612466390692691
ExtraTreesRegressor 의 정답률 =  0.7439679639434615
GaussianProcessRegressor 의 정답률 =  -11579.198096294853
GradientBoostingRegressor 의 정답률 =  0.7712366162759282
HuberRegressor 의 정답률 =  0.624941716194424
KNeighborsRegressor 의 정답률 =  0.7312345084452816
KernelRidge 의 정답률 =  -2.1073591596976247
Lars 의 정답률 =  0.6268878227652545
LarsCV 의 정답률 =  0.6268878227652545
Lasso 의 정답률 =  0.13344528372728037
LassoCV 의 정답률 =  0.6324752081132299
LassoLars 의 정답률 =  -0.005510413547094695
LassoLarsCV 의 정답률 =  0.6268878227652545
LassoLarsIC 의 정답률 =  0.6268878227652545
LinearRegression 의 정답률 =  0.6268878227652543
LinearSVR 의 정답률 =  -3.2380807247967853
MLPRegressor 의 정답률 =  0.41295908156958105
'''


