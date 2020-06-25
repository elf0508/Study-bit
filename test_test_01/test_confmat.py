# 혼동 행렬 구현하기
# 실제 혼동행렬의 각 성분의 개수 

import numpy as np
from sklearn.metrics import confusion_matrix

# 0은 양성, 1은 음성을 나타낸다.

y_true = [0,0,0,1,1,1]
y_pred = [1,0,0,1,1,1]

# 변수 confmat에 y_true, y_pred의 혼동 행렬을 저장
confmat = confusion_matrix(y_true, y_pred)

print(confmat)

# [[2 1]
#  [0 3]]

#############################

# precision_score : 적합률(정밀도)
# recall_score : 재현률
# f1_score : F값

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

y_true = [0,0,0,1,1,1]
y_pred = [1,0,0,1,1,1]

# %.3f --> 소수점 셋째자리까지 표시

print("Precision : %.3f" % precision_score(y_true, y_pred))  
print("Recall : %.3f" % recall_score(y_true, y_pred))
print("F1 : %.3f" % f1_score(y_true, y_pred))

# Precision : 0.750
# Recall : 1.000
# F1 : 0.857




