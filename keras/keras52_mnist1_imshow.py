import numpy as np
import matplotlib.pyplot as plt 

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 값을 찾아서,왼쪽으로 반환한다.

print(x_train[0])  # 0 ~ 255
print('y_train : ', y_train[0])  # 5

print(x_train.shape) # (60000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)

print(y_train.shape) # (60000,)  6만개의 스칼라를 가진, 벡터 1개
print(y_test.shape)  # (10000,)  1만개의 스칼라를 가진, 벡터 1개

# print(x_train[0].shape)  # (가로 : 28, 세로 : 28)
# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# plt.show()

print(x_train[0].shape)  # (가로 : 28, 세로 : 28)
plt.imshow(x_train[1], 'gray')
# plt.imshow(x_train[0])
plt.show()

# print(x_train[0].shape)  # (가로 : 28, 세로 : 28)
# plt.imshow(x_train[1], 'gray')
# # plt.imshow(x_train[0])
# plt.show()

# print(x_train[0].shape)  # (가로 : 28, 세로 : 28)
# plt.imshow(x_train[59999], 'gray')
# # plt.imshow(x_train[0])
# plt.show()




