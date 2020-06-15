
# 이미지 데이터 처리

from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
# from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.layers import Flatten, Convolution3D, MaxPooling3D
from keras.models import load_model
# from keras.layers import Dense, LSTM, Dropout, Conv2D
from keras.layers import Dense, LSTM, Dropout, Conv3D


### 이미지 파일 불러오기 및 카테고리 정의
# caltech_dir = './project/mini/images'
caltech_dir = 'D:/Study-bit/project_mini/img'
categories = ['dog_open', 'dog_closed']
# categories = ['scooter', 'supersports', 'multipurpose', 'cruiser']
nb_classes = len(categories)

### 가로, 세로, 채널 쉐이프 정의
image_w = 64
image_h = 64
pixels = image_h * image_w * 3

### 이미지 파일 Data화
X = []
Y = []

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = caltech_dir + '/' + cat
    files = glob.glob(image_dir + "/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

        if i % 700 == 0:
            print(cat, ':', f)

x = np.array(X)
y = np.array(Y)

'''
enumerate : 열거하다
리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능을 가집니다.
이 함수는 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 
인덱스 값을 포함하는 enumerate 객체를 리턴합니다.
보통 enumerate 함수는 for문과 함께 자주 사용됩니다.

dog_open : 50개

dog_closed : 50개
'''

print(x.shape) # (200, 100, 100, 3)
print(y.shape) # (100, 2)

# print(x.shape) # (100, 64, 64, 3)
# print(y.shape) # (200, 4)


### 데이터 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
xy = (x_train, x_test, y_train, y_test)

print(x_train.shape)    # (80, 64, 64, 3)
print(x_test.shape)     # (20, 64, 64, 3)
print(y_train.shape)    # (80, 2)
print(y_test.shape)     # (20, 2)

# print(x_train.shape)    # (160, 100, 100, 3)
# print(x_test.shape)     # (40, 100, 100, 3)
# print(y_train.shape)    # (160, 4)
# print(y_test.shape)     # (40, 4)

### 데이터 SAVE

np.save('D:/Study-bit/project_mini/data.npy', xy)
print('ok', len(y))

cv2.waitKey(0)
# X.append(img/256)
# Y.append(label)

np.save('./project_mini/data/multi_image_data.npy', xy)
# print('ok', len(y))

### 데이터 load

# X_train, X_test, Y_train, Y_test = np.load('D:/Study-bit/project_mini/data.npy')
xy = np.load('D:/Study-bit/project_mini/data.npy', allow_pickle=True)

print(xy)

# x : scaler
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x).reshape(64, 8, 8, 3)

# print(x.reshape)


### 모델 만들기

model = Sequential()
model.add(Conv3D(10,(2, 2), input_shape =(8, 8, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(MaxPooling3D(pool_size = 2))

model.add(Conv3D(50,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv3D(80,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv3D(100, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv3D(150, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv3D(120, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv3D(80,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv3D(40,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv3D(20,(2, 2),activation = 'relu', padding = 'same'))
model.add(Flatten())
model.add(Dense(3, activation = 'sigmoid'))

model.summary()

'''
model = Sequential()
model.add(Conv2D(10,(3, 3), input_shape =(8, 8, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(50,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(80,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(100, (3, 3),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(150, (3, 3),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(120, (3, 3),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(80,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(40,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(20,(3, 3),activation = 'relu', padding = 'same'))
model.add(Flatten())
model.add(Dense(3, activation = 'sigmoid'))

model.summary()

# callbacks
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
# earlystopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1 )
# tensorboard
ts_board = TensorBoard(log_dir = 'graph', histogram_freq = 0,
                        write_graph = True, write_images = True)
# modelcheckpotin
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
ckpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                          save_best_only = True)



#3. compile, fit
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
                validation_split = 0.2, verbose = 2,
                callbacks = [es])



#4. evaluate, predict

loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss: ', loss )
print('acc: ', acc)

# graph
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], c= 'red', marker = '^', label = 'loss')
plt.plot(hist.history['val_loss'], c= 'cyan', marker = '^', label = 'val_loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], c= 'red', marker = 'o', label = 'acc')
plt.plot(hist.history['val_acc'], c= 'cyan', marker = 'o', label = 'val_acc')
plt.title('accuarcy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.show()
'''
