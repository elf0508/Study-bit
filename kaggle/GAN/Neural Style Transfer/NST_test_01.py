'''

https://www.popit.kr/neural-style-transfer-%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0/

https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko

이미지 콘텐츠를 특정 스타일에 맞춰 최적화시키는 기존의 스타일 전이 알고리즘

'''
import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

import pandas as pd
import os

from glob import glob
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import scipy as sp

def tensor_to_image(tensor):
      tensor = tensor*255
      tensor = np.array(tensor, dtype=np.uint8)

      if np.ndim(tensor)>3:
         assert tensor.shape[0] == 1
         tensor = tensor[0]
  
      return PIL.Image.fromarray(tensor)


content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg

style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


# 이미지를 불러오는 함수를 정의하고, 최대 이미지 크기를 512개의 픽셀로 제한한다.

def load_img(path_to_img):
    max_dim = 512

    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img) [:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


# 이미지를 출력하기 위한 간단한 함수를 정의

def imshow(image, title = None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis = 0)

    plt.imshow(image)
    if title:
        plt.title(title)


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
# plt.show()

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
# plt.show()


#  TF-Hub를 통한 빠른 스타일 전이

import tensorflow_hub as hub     #  pip install tensorflow-hub

# model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")

# embeddings = model(["The rain in Spain.", "falls",
#                       "mainly", "In the plain!"])

# print(embeddings.shape)   # (4, 128)


hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

style_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

tensor_to_image(style_image)

plt.subplot(1, 2, 1)
imshow(style_image, 'Style Image')
# plt.show()


# 모델 : VGG19 사용

x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)

x = tf.image.resize(x, (244, 224))

vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')

prediction_probabilities = vgg(x)

prediction_probabilities.shape

# 분류층을 재외한 VGG19 모델을 불러오고, 각 층의 이름을 출력

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

# print(prediction_probabilities)

vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')

print()

for layer in vgg.layers:
    print(layer.name)

'''
input_2
block1_conv1
block1_conv2
block1_pool
block2_conv1
block2_conv2
block2_pool
block3_conv1
block3_conv2
block3_conv4
block3_pool
block4_conv1
block4_conv2
block4_conv3
block4_conv4
block4_pool
block5_conv1
block5_conv2
block5_conv3
block5_conv4
block5_pool
block1_conv1

'''

# 이미지의 스타일과 콘텐츠를 나타내기 위한 모델의 중간층들을 선택

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)

num_style_layers = len(style_layers)   

######################################

model = Model()
model = Model(inputs, outputs)


""" 중간층의 출력값을 배열로 반환하는 vgg 모델을 만듭니다."""

# 이미지넷 데이터셋에 사전학습된 VGG 모델을 불러옵니다

def vgg_layers(layer_names):
   
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)

  return model

# 위 함수를 이용해 모델을 만들기

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

# 각 층의 출력에 대한 통계량

for name, output in zip(style_layers, style_outputs):

  print(name)
  print("  크기: ", output.numpy().shape)
  print("  최솟값: ", output.numpy().min())
  print("  최댓값: ", output.numpy().max())
  print("  평균: ", output.numpy().mean())
  print()

'''
block1_conv1
  크기:  (1, 424, 512, 64)
  최솟값:  0.0
  최댓값:  775.3464
  평균:  31.370852

block2_conv1
  크기:  (1, 212, 256, 128)
  최솟값:  0.0
  최댓값:  3173.9556
  평균:  180.7034

block3_conv1
  크기:  (1, 106, 128, 256)
  최솟값:  0.0
  최댓값:  10159.233
  평균:  201.40997

block4_conv1
  크기:  (1, 53, 64, 512)
  최솟값:  0.0
  최댓값:  20731.262
  평균:  684.8376

block5_conv1
  크기:  (1, 26, 32, 512)
  최솟값:  0.0
  최댓값:  3009.3987
  평균:  46.61009

'''
##########################
'''

스타일 계산하기

이미지의 콘텐츠는 중간층들의 특성 맵(feature map)의 값들로 표현된다.

이미지의 스타일은 각 특성 맵의 평균과 피쳐맵들 사이의 상관관계로 설명할 수 있다. 
이런 정보를 담고 있는 그람 행렬(Gram matrix)은 각 위치에서 특성 벡터(feature vector)끼리의 외적을 구한 후,
평균값을 냄으로써 구할 수 있다.

'''

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd' , input_tensor, input_tensor)

    input_shape = tf.shape(input_tensor)

    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return result / (num_locations)


# 스타일과 콘텐츠 추출하기
# 스타일과 콘텐츠 텐서를 반환하는 모델을 만든다.

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = len(style_layers)
        self.vgg.trainable = False


    def call(self, inputs):
        "[0, 1] 사이의 실수 값을 입력으로 받는다."
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])


        style_outputs = [gram_matrix(style_outputs)
                            for style_outputs in style_outputs]

        content_dict = {content_name : value
                        for content_name, value
                        in zip(self.style_layers, style_outputs)}

        style_dict = {style_name : value
                        for style_name, value
                        in zip(self.style_layers, style_outputs)}

        return {'content' : content_dict, 'style' : style_dict}

# 이미지가 입력으로 주어졌을때, 이 모델은 style_layers의 스타일과 content_layers의 
# 콘텐츠에 대한 그람 행렬을 출력한다:

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('스타일 : ')

for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    크기 : ", output.numpy().shape)
    print("    최솟값 : ", output.numpy().min())
    print("    최댓값 : ", output.numpy().max())
    print("    평균 : ", output.numpy().mean())

    print()

print("콘텐츠 : ")

for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    크기 : ", output.numpy().shape)
    print("    최솟값 : ", output.numpy().min())
    print("    최댓값 : ", output.numpy().max())
    print("    평균 : ", output.numpy().mean())

    print()


#########################################

'''
경사하강법 실행

이제 스타일과 콘텐츠 추출기를 사용해 스타일 전이 알고리즘을 구현한다. 
타깃에 대한 입력 이미지의 평균 제곱 오차를 계산한 후, 오차값들의 가중합을 구한다.

'''

# 스타일과 콘텐츠의 타깃값을 지정한다

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# 최적화시킬 이미지를 담을 tf.Variable을 정의하고 콘텐츠 이미지로 초기화한다. 
# (이때 tf.Variable는 콘텐츠 이미지와 크기가 같아야 한다.

image = tf.Variable(content_image)

# 픽셀 값이 실수이므로 0과 1 사이로 클리핑하는 함수를 정의

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# 옵티마이저를 생성. LBFGS를 추천하지만, Adam도 충분히 적합하다.

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


# 최적화를 진행하기 위해, 전체 오차를 콘텐츠와 스타일 오차의 가중합으로 정의

style_weight=1e-2
content_weight=1e4


def style_content_loss(outputs):

    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    
    return loss


# 이미지 업데이트

@tf.function()

def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


train_step(image)
train_step(image)
train_step(image)
tensor_to_image(image)


import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0

for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)

    print(".", end='')

  display.clear_output(wait=True)
  display.display(tensor_to_image(image))

  print("훈련 스텝: {}".format(step))
  
end = time.time()
print("전체 소요 시간: {:.1f}".format(end-start))


'''
총 변위 손실

이 기본 구현 방식의 한 가지 단점은 많은 고주파 아티팩(high frequency artifact)가 생겨난다는 점이다. 

아티팩 생성을 줄이기 위해서는 이미지의 고주파 구성 요소에 대한 레귤러리제이션(regularization) 항을 추가해야 한다.
스타일 전이에서는 이 변형된 오차값을 총 변위 손실(total variation loss)라고 한다

'''

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var


x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)

imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")
plt.show()

plt.subplot(2,2,2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")
plt.show()

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2,2,3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")
plt.show()

plt.subplot(2,2,4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")
plt.show()

# 위 코드를 이용하면, 위 이미지들은 고주파 구성 요소가 늘어났다는 것을 보여준다.
# 고주파 구성 요소가 경계선 탐지기의 일종이다. 
# 이를테면 소벨 경계선 탐지기(Sobel edge detector)를 사용하면 유사한 출력을 얻을 수 있다

plt.figure(figsize=(14,10))

sobel = tf.image.sobel_edges(content_image)

plt.subplot(1,2,1)
imshow(clip_0_1(sobel[...,0]/4+0.5), "Horizontal Sobel-edges")
plt.show()

plt.subplot(1,2,2)
imshow(clip_0_1(sobel[...,1]/4+0.5), "Vertical Sobel-edges")
plt.show()

# 정규화 오차는 각 값의 절대값의 합으로 표현된다.

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)

    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

total_variation_loss(image).numpy()


tf.image.total_variation(image).numpy()

# 다시 최적화하기
# 가중치 정의


total_variation_weight = 30

@tf.function()

def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# 최적화할 변수를 다시 초기화한다

image = tf.Variable(content_image)


# 최적화를 수행

import time
from keras.engine.training import Model
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0

for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)

    print(".", end='')

  display.clear_output(wait=True)
  display.display(tensor_to_image(image))

  print("훈련 스텝: {}".format(step))

end = time.time()
print("전체 소요 시간: {:.1f}".format(end-start))


