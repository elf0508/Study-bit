# vgg16 (레이어가 16개 라는 뜻)
# 이미지 분석

from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

# vgg16 = VGG16()  # (None, 224, 224, 3)

# vgg16.summary()    # VGG16 다운로드

# take_model = VGG19()
# take_model.summary()

# take_model = Xception()
# take_model = ResNet101()
# take_model = ResNet101V2()
# take_model = ResNet152()
# take_model = ResNet152V2()
# take_model = ResNet50()
# take_model = ResNet50V2()
# take_model = InceptionV3()
# take_model = InceptionResNetV2()
# take_model = MobileNet()
# take_model = MobileNetV2()
# take_model = DenseNet121()
# take_model = DenseNet169()
# take_model = DenseNet201()
# take_model = NASNetLarge()
# take_model = NASNetMobile()

applications = [VGG19, Xception, ResNet101, ResNet101V2, ResNet152,ResNet152V2, ResNet50, 
                ResNet50V2, InceptionV3, InceptionResNetV2,MobileNet, MobileNetV2, 
                DenseNet121, DenseNet169, DenseNet201]

for i in applications:
    take_model = i()

    model = VGG19()
    model = Xception()
    model =  ResNet101()
    model = ResNet101V2()
    model = ResNet152()
    model = ResNet152V2()
    model = ResNet50()
    model = ResNet50V2()
    model = InceptionV3()
    model = InceptionResNetV2()
    model = MobileNet()
    model = MobileNetV2()
    model = DenseNet121()
    model = DenseNet169()
    model = DenseNet201()
    model = NASNetLarge()
    model = NASNetMobile()


vgg16 = VGG16()
vgg16 = VGG16(include_top = False)

model = Sequential()
model.add(vgg16)

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

'''
# VGG-16 모델은 Image Net Challenge에서 Top-5 테스트 정확도를 92.7% 달성

from keras.applications import VGG16, VGG19
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

vgg16 = VGG16()
vgg19 = VGG19()

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# model = Sequential()
# model.add(vgg19)
# # model.add(Flatten())
# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()
'''








