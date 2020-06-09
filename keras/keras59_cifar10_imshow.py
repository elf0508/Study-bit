# 10가지 이미지를 찾아서, 칼라(3)

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)  # (50000, 32, 32, 3)
print(x_test.shape)   # (10000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
print(y_test.shape)   # (10000, 1)

plt.imshow(x_train[0])
plt.show()

