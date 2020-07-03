import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Input, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils
# from keras.datasets import cifar100
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

(x_train, y_train),(x_test, y_test) = cifar10.load_data()


ResNet101 = ResNet101(include_top = False, input_shape = (32, 32, 3))

# vgg16.summary()

model = Sequential()
model.add(ResNet101)
model.add(Flatten())
model.add(Dense(256))      #
model.add(BatchNormalization())    #
model.add(Activation('relu'))      #
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy',         # 원핫인코딩을 하지 않았을 때, 다중분류 손실함수
              metrics = ['accuracy'],                           # sparse는 개인 취향이다!
              optimizer = Adam(1e-4))                           # 0.0001
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 32,
                 validation_split = 0.3, verbose = 1)

res = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", res[0])
print("acc : ", res[1])

# 시각화
plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.title('loss')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['accuracy'], marker = '.', c = 'violet', label = 'acc')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'green', label = 'val_acc')
plt.title('accuracy')
plt.grid()
plt.ylim(0, 1.0)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'lower right')
plt.show()

# loss :  2.4153740846633913
# acc :  0.7562000155448914