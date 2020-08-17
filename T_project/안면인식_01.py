import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import csv

#Load-------------------------------------------------------------------------------------------------------------------

def load_images_from_folder(folder):
    images = []

    for filename in tqdm(folder):
        img = cv2.imread(filename,cv2.IMREAD_COLOR) 
        img = img[15:200,30:300] 
        images.append(img)

    images = np.array(images)

    return images

#
def globb(b):
    for i in b:
        x = glob(i)

        return x


a=[]
def load(t):
    for root, dirs, files in os.walk(t):
        for fname in dirs:
            full_dirs = os.path.join(root, fname)

            full = glob('{}/*.jpg'.format(full_dirs))
            if full != []:

                T = full
                # if len(T)!=20:
                #     print(T)
                try:
                    u = load_images_from_folder(T)
                    a.append(u)
                except:
                    print(T)
                    del T

    u = np.asarray(a)

    return u

x = load('D:/Study-bit/T_project/H_Resolution/koreaface1/e')
# x = load('D:/s/koreaface/Low_Resolution/koreaface1/e')

np.save('D:/Study-bit/T_project/H_Resolution/koreaface/xe.npy', arr = x)

print(x.shape)   # (540, 20, 185, 270, 3)