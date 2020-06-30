'''
n_samples : 준비할 데이터 수

n_classes : 클래스 수, 지정하지 않으면 2로 지정된다.

n_features : 데이터의 특징량 수

n_redundant : 분류에 불필요한 특징량(여분의 특징량) 수

random_state : 난수시드(난수 패턴을 결정하는 요소)

'''

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier 

x, y = make_classification(n_samples = 50, n_features = 2, n_redundant = 0, random_state = 0 )

