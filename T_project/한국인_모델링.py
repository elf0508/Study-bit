import dlib, cv2 

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib.patheffects as path_effects

####

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import os

from tqdm import tqdm

from glob import glob

from sklearn.model_selection import train_test_split

import csv

# 데이터 가져오기

x = np.load('D:/Study-bit/T_project/Low_Resolution/Image1.npy')


print(x.shape)   # (198765, 105, 130, 3)

# print(y.shape)


# 모델

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

model = Sequential()

model.add(Conv2D(10, (2, 2), input_shape = (105, 130, 3 ), activation = 'relu', padding = 'same'))
model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(3, activation = 'softmax'))

model.summary() 


# callbacks 

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# earlystopping

es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose =1)

# Tensorboard

ts_board = TensorBoard(log_dir = 'graph', histogram_freq= 0,
                      write_graph = True, write_images=True)
# Checkpoint

# modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'

# ckecpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
#                             save_best_only= True)


#3. compile, fit

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x, epochs =100, batch_size= 64,
                validation_split = 0.2, verbose = 2,
                callbacks = [es])



# evaluate

loss, acc = model.evaluate(x, batch_size = 64)

print('loss: ', loss )
print('acc: ', acc)
