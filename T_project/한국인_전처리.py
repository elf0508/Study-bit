

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import csv
import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generator=ImageDataGenerator(
    shear_range=0.5,
    width_shift_range=0.10,
    height_shift_range=0.10
)

augment_size =3

# C:\Users\bitcamp\Desktop\ImageClassification_DjangoApp-master

def load_images_from_folder(filename):

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = img[15:, 35:135] # 컬러사진을 이용한다.
    img = img.reshape(1,100,100,3)

    randix = np.random.randint(img.shape[0], size=augment_size)
    x_augmented = img[randix].copy()

    image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
    x = np.concatenate([img, x_augmented], axis=0)

    return x

load_images_from_folder("T_project/Low_Resolution/19062421/S001/L1/E01/C7.jpg")
# load_images_from_folder("C:/Users/bitcamp/Downloads/Low_Resolution/19082721/S001/L1/E01/C7.jpg")

def search(dirname):

    filenames = os.listdir(dirname)

    b=[]
    v=[]
    n=[]
    x=[]
    k=[]
    t=[]

    Image = []

    for i in tqdm(filenames):

        k.append(os.path.join('{}'.format(dirname), i))


    for j in k:

        filenames=os.listdir(j)

        for i in filenames:
            if i == 'S001':
                x.append(os.path.join('{}'.format(j), i))
            if i == 'S002':
                x.append(os.path.join('{}'.format(j), i))
            if i == 'S003':
                x.append(os.path.join('{}'.format(j), i))

            else:
                continue




    for dirname2 in x:

        l = os.listdir('{}'.format(dirname2))

        for i in l:
            if i == 'L1':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L2':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L3':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L4':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L8':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L9':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L19':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L20':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L22':
                b.append(os.path.join('{}'.format(dirname2), i))
            elif i == 'L25':
                b.append(os.path.join('{}'.format(dirname2), i))
            else:
                continue

    for dirname3 in b:

        e = os.listdir('{}'.format(dirname3))

        for i in e:
            if i == 'E01':
                v.append(os.path.join('{}'.format(dirname3), i))
            elif i == 'E02':
                v.append(os.path.join('{}'.format(dirname3), i))
            elif i == 'E03':
                v.append(os.path.join('{}'.format(dirname3), i))
            else:
                continue

    for dirname4 in v:

        c = os.listdir('{}'.format(dirname4))

        for i in tqdm(c):

            if i == 'C7.jpg':

                n.append(load_images_from_folder(os.path.join('{}'.format(dirname4), i)))

            else:
                continue


    f = np.array(n)

    return f



x=search('filenames = os.listdir(dirname)')
# x=search('D:/Study-bit/T_project/Low_Resolution')
# x=search('C:/Users/bitcamp/Downloads/Low_Resolution')

print(x.shape)

np.save('T_project/Low_Resolution/한국인안면/Imageall.npy', arr=x)
# np.save('D:/s/한국인안면/Imageall.npy', arr=x)
