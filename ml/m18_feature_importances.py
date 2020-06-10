# feature_importances
#유방암 - 컬럼 30개
# 이진분류

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train,y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 42
)

model = DecisionTreeClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

'''
[0.         0.         0.         0.        
 0.         0.
 0.         0.70458252 0.         0.        
 0.01221069 0.
 0.         0.         0.         0.        
 0.02530295 0.0162341
 0.         0.         0.05329492 0.05959094 0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]

 acc :  0.9298245614035088
'''

import matplotlib.pyplot as plt
import numpy as np
# 시각화 패키지 소개 : https://datascienceschool.net/view-notebook/d0b1637803754bb083b5722c9f2209d0/

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


# n_estimators : 생성할 tree의 개수 
# max_features : 최대 선택할 특성의 수
# max_features를 n_features로 설정하면 -> 트리 각 분기에서 모든 특성 고려하므로 무작위성 들어가지 않는다.
# 하지만 부스트 랩 샘플링으로 인한 무작위성은 그대로입니다.
# max_features = 1 -> 트리의 분기는 테스트할 특성을 고를 필요가 없게 되며 
# 그냥 무작위로 선택한 특성의 임계치를 찾기만 하면 된다. 

# 결국 max_features 값을 크게 하면 랜덤 포레스트의 트리들은 매우 비슷
#  -> 가장 두드러진 특성을 이용해 데이터에 잘 맞춤

# max_features 낮추면  랜덤 포레스트 트리들이 많이 달라지고 각 트리는 맞추기 위해 깊이가 깊어지게 된다.
# 랜덤포레스트 예측할 때는 먼저 알고리즘이 모델에 있는 모든 트리의 예측 만든다.
# 회귀의 경우: 이 예측들을 평균하여 최종 예측을 만든다. 
# 분류의 경우: 약한 투표 전략을 사용한다 
# 즉) 각 알고리즘이 가능성 있는 출력 레이블의 확률을 제공합으로써 간접 예측 트리들이 
# 예측한 확률을 평균 내어 가장 높은 확률을 가진 클래스가 예측값이 된다. 


'''
n_feature = cancer.data.shape[1]

index = np.arange(n_feature)

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)

forest.fit(x_train, y_train)

plt.barh(index, forest.feature_importances_, align='center')

plt.yticks(index, cancer.feature_names)

plt.ylim(-1, n_feature)

plt.xlabel('feature importance', size=15)

plt.ylabel('feature', size=15)

plt.show()
'''
'''
# 특성 중요도 시각화 

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
'''