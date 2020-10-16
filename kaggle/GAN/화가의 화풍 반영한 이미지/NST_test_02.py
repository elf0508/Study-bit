#  https://www.youtube.com/watch?v=h1U52D_YOxY&list=WL&index=36

import tensorflow as tf

from google.colab import files  

uploaded = file.upload()

for fn in uploaded.keys():
    print('user uploaded file "{name}" wiht length {length} bytes'.format(name = fn, length = len(uploaded[fn])))

import keras

from keras.preprocessing.image import load_img, img_to_array, save_img

# 변환하려는 이미지 경로
target_image_path = 'D:/Study-bit/kaggle/GAN/화가의 화풍 반영한 이미지/content'

# 스타일 이미지 경로
style_reterence_image_path = 'D:/Study-bit/kaggle/GAN/화가의 화풍 반영한 이미지/style'

# 생성된 사진의 차원
width, heigth = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / heigth)  #  비율에 맞춰 줄여준다

import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img = load_img(image_path, target_size = (img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img

def deprocess_image(x):
    # vgg19.preprocess_input 함수에서 일어나는 변환을 복원하기 위해
    # imageNet의 평균 픽셀 값을 더하고

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR' --> 'RGB' 로 변환한다

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x


from keras import backend as K

# 이미 준비되어있는 이미지 이므로 constant를 이용해서 정의
target_image = K.constant(preprocess_image(target_image_path))

style_reterence_image = K.constant(preprocess_image(style_reterence_image_path))

# 생성된 이미지를 담을 플레이스홀터
combination_image = K.placeholder((1, img_height, img_width, 3))

# 세 개의 이미지를 하나의 배치로 합친다
input_tensor = K.concatenate([target_image,
                                style_reterence_image,
                                combination_image], axis=0)


# 세 이미지의 배치를 입력으로 받는 VGG 네트워크를 만든다
# 이 모델은 사전 훈련된 imageNet 가중치를 로드한다

model = vgg19.VGG19(input_tensor = input_tensor,
                        width = 'imagenet',
                        include_top = False)

print('모델 로드 완료.')




