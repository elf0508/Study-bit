# 49-2번의 첫번째 답

# x = [1,2,3]
# x = x -1
# print(x)


import numpy as np

'''
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1
print(y)

from keras.utils import np_utils
y = np_utils.to_categorical(y)  # 시작이 0 부터(자동으로 넣는다.)
print(y)
print(y.shape)

y_pred = np.argmax(y, axis =1)                 # axis =1 , 행별로 최댓값을 뺀다.
print(y_pred)                                  # y = [0 1 2 3 4 0 1 2 3 4]
y1_pred = np.argmax(y, axis=1 )+1 
print(y1_pred)
'''

# 2번의 두번째 답

y = np.array([1,2,3,4,5,1,2,3,4,5])

print(y.shape)  # (10, )
# y = y.reshape(-1, 1)
y = y.reshape(10, 1)  # 2차원

from sklearn.preprocessing import OneHotEncoder  # shape를 맞춰줘야 한다.

aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()

'''
[[1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''

print(y)

print(y.shape)  # (10, 5)