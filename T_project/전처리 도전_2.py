import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shutil import copyfile
import os
import shutil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, UpSampling2D


# base_dir = 'D:/Study-bit/T_project'
base_dir = 'D:/Study-bit/T_project/img'

img_dir = 'D:/Study-bit/T_project/img'

print(len(os.listdir(img_dir)))      # 들어있는 이미지 총 60개

print(os.listdir(img_dir)[:10])      
# ['C1.jpg', 'C10.jpg', 'C11.jpg', 'C12.jpg', 'C13.jpg', 
# 'C14.jpg', 'C15.jpg', 'C16.jpg', 'C17.jpg', 'C18.jpg']  10개만 랜덤으로 가져오기

# 30개의 이미지만 샘플로 선별해서 다른 폴더로 복사해보기
imgs30_dir = os.path.join(base_dir, 'imgs30_dir')
# imgs30_dir = os.path.join(base_dir, 'C30')

# os.mkdir(imgs30_dir)


fnames = ['img.{}.jpg'.format(i) for i in range(30)]

for fname in fnames:
    
    src = os.path.join(img_dir, fname)

    dst = os.path.join(imgs30_dir, fname)

    # shutil.copyfile(src, dst)

    print(os.listdir(imgs30_dir))

#######################################################

# 이미지 파일을 로딩, float array로 변환 후 전처리

# from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

img_name = 'img.10.jpg'

# scriptpath = os.path.dirname(img_name)

# img_path = os.path.join(scriptpath, img_name)
img_path = os.path.join(imgs30_dir, img_name)

img = image.load_img(img_path, target_size=(250, 250))

img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)

print(img_tensor.shape)

img_tensor /= 255.


print(img_tensor[0])

#  한개의 이미지 파일의 array 를 시각화하기

plt.reParams['figure.figsize'] = (10, 10)

plt.show(img_tensor[0])

plt.show()

# 30개의 이미지 데이터를 6*5 격자에 나누어서 시각화하기


def preprocess_img(img_path, target_size=100):

     from keras.preprocessing import image

     

     img = image.load_img(img_path, target_size=(target_size, target_size))

     img_tensor = image.img_to_array(img)

    
     img_tensor = np.expand_dims(img_tensor, axis=0)

    
     img_tensor /= 255.

     
     return img_tensor


n_pic = 30

n_col = 5

n_row = int(np.ceil(n_pic / n_col))

target_size = 100

margin = 3

total = np.zeros((n_row * target_size + (n_row - 1) * margin, n_col * target_size + (n_col - 1) * margin, 3))


img_seq = 0


for i in range(n_row):

    for j in range(n_col):


        fname = 'cat.{}.jpg'.format(img_seq)

        img_path = os.path.join(cats30_dir, fname)


        img_tensor = preprocess_img(img_path, target_size)


        horizontal_start = i * target_size + i * margin

        horizontal_end = horizontal_start + target_size

        vertical_start = j * target_size + j * margin

        vertical_end = vertical_start + target_size


        total[horizontal_start : horizontal_end, vertical_start : vertical_end, :] = img_tensor[0]

        
        img_seq += 1


plt.figure(figsize=(200, 200))

plt.imshow(total)

plt.show()






