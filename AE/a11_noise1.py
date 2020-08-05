# PCA : 선형
# 오토인코더 (autoencoder) : 비선형

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def autoencoder(hidden_layer_size):
    
    model = Sequential()

    model.add(Dense(units = hidden_layer_size, input_shape = (784, ), activation = 'relu'))

    model.add(Dense(units = 784, activation = 'sigmoid'))

    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()

x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
#                           60000       ,            784
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

x_train = x_train/255.
x_test = x_test/255.

# noise 추가

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
# x_train_noised = x_train + np.random.normal(0, 0.5, size=x_train.shape)
# x_test_noised = x_test + np.random.normal(0, 0.5, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


model = autoencoder(hidden_layer_size = 154)
# model = autoencoder(hidden_layer_size = 32)

# model.compile(optimize = 'adam', loss = 'mse', metrics = ['acc'])
#  loss: 0.0104 - acc: 0.0108

model.compile(optimize = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
# loss: 0.0941 - acc: 0.8141

model.fit(x_train_noised, x_train, epochs = 20, batch_size = 128) 
# model.fit(x_train_noised, x_train_noised, epochs = 10) # <-- 노이즈 제거 훈련 부분

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9,ax10),
        (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다 / 잡음 없다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    # ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음 있다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap = 'gray')
    # ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('NOISE', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 마지막에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
   
plt.show()