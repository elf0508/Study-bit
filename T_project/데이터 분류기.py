'''
https://bskyvision.com/749

(각 20장씩, 3개의 파일)  데이터셋이 준비되었으니 이제 분류기를 만들어서 훈련시작. 

대개 훈련셋, 검증셋, 훈련셋으로 나눠서 제대로 훈련이 되었는지 평가하지만, 
편의상 모든 이미지를 훈련에 투입. 

이미지넷에서 미리 훈련된 ResNet50 모델을 가져와서 
최종 출력 레이어를 제거한 후에 global average pooling 레이어, 3개의 뉴런으로 구성된 FC 레이어를 추가. 

세 가지를 분류하면 되기 때문에 3개의 뉴런을 갖도록 설정했습니다. 

그 다음에 이미지들과 그에 맞는 레이블로 모델을 훈련시킵니다. 

이때 중요한 것은 각 이미지 폴더에서 이미지를 불러온 후에 이미지들을 한번 섞어준다. 

안 그러면 훈련시 같은 이미지가 연달아서 나오기 때문에 

모델의 가중치를 제대로 훈련시킬 수 없다. 

훈련이 완료되면 훈련된 모델을 model.h5로 저장. 


'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
 
 
path_dir1 = 'T_project/file/L4'
path_dir2 = 'T_project/file/L8'
path_dir3 = 'T_project/file/L8_2'
 
file_list1 = os.listdir(path_dir1) # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
file_list3 = os.listdir(path_dir3)
 
# train용 이미지 준비
num = 0;
train_img = np.float32(np.zeros((1268, 224, 224, 3))) # 394+413+461
train_label = np.float64(np.zeros((1268, 1)))
 
for img_name in file_list1:
    img_path = path_dir1+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1
 
for img_name in file_list2:
    img_path = path_dir2+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 1 # paper
    num = num + 1
 
for img_name in file_list3:
    img_path = path_dir3+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 2 # scissors
    num = num + 1
 
 
# 이미지 섞기
     
n_elem = train_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
train_label = train_label[indices]
train_img = train_img[indices]
  
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)
 
base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()

print("Number of layers in the base model: ", len(base_model.layers))
 
GAP_layer = GlobalAveragePooling2D()
dense_layer = Dense(3, activation=tf.nn.softmax)
 
model = Sequential([
        base_model,
        GAP_layer,
        dense_layer
        ])
 
base_learning_rate = 0.001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
 
model.fit(train_img, train_label, epochs=5)
 
# save model
# model.save("model.h5")
 
print("Saved model to disk")  