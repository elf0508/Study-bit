# 회귀분류
# 1.회귀 : score 와 R2 비교
# 2.분류 : score 와 accuracy_score 비교 

from sklearn.datasets import load_boston
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, train_size =0.2, shuffle=True)

scaler = StandardScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x = scaler.transform(x)

print(x.shape)
print(y.shape)

# 2. 모델
# model =SVC()           # 원 핫 인코딩 필요 없음
# model = LinearSVC()                                       
# model = KNeighborsClassifier(n_neighbors = 1)
# model =KNeighborsForestClassifier()
model = RandomForestRegressor()
# model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, r2_score

# acc = accuracy_score(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("score : ", score )
# print("acc : ", acc )
print("R2 : ", R2 )

# score :  0.7832508821947954
# R2 :  0.7832508821947954




