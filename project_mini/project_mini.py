
# 이미지 데이터 처리

from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
from keras.layers import Dense, LSTM, Dropout, Conv2D
from keras.preprocessing.image import ImageDataGenerator
import sklearn.metrics as metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

### 이미지 파일 불러오기 및 카테고리 정의

caltech_dir = 'D:/Study-bit/project_mini/img'
# 디렉토리 지정
# caltech_dir = 'train 이미지 경로'

categories = ['dog_1', 'dog_2']
# 카테고리 분류
nb_classes = len(categories)  # 위에 categories에 [  ] 2개가 들어가 있어서 len = 2

### 가로, 세로, 채널 쉐이프 정의

image_w = 64
image_h = 64
pixels = image_h * image_w * 3

# 사진의 크기를 64*64 크기로 변환 

### 이미지 파일 Data화 (이미지 파일 변환)
# numpy로 데이터화
X = []
Y = []

for idx, cat in enumerate(categories):  
    # 카테고리별로 돌면서 0으로 초기화
    # 카테고리 안에 있는 것을 밖으로 뽑아주는 것(인덱스, 파일명)
    label = [0 for i in range(nb_classes)]  # == [0 for i in range(2)] --> label [0, 1]
    # print(label)
    label[idx] = 1  # 인덱스 자리에 1을 넣어주겠다. --> label [1, 0], [0, 1] : y를 만들어 주는 것
    image_dir = caltech_dir + '/' + cat  # 이미지 디렉토리 지정, cat --> category
    files = glob.glob(image_dir + "/*.jpg") # glob.glob : 이미지를 불러오는 함수, jpg 파일형식으로 
                            # + "/** " : 모든 파일을 가져오겠다는 뜻
    print(cat, " 파일 길이 : ", len(files))
    #      파일명,             가져온 파일의 갯수
    # 여기까지가 cat으로 지정한 폴더를  불러오겠다는 뜻

    for i, f in enumerate(files): # 파일(이미지)를 불러온다. i : 파일 번호, 파일 이름 / f : 파일
        img = Image.open(f) 
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))  # resize (크기 조정)
        data = np.asarray(img)  # 이미지를 np형식으로 바꿔준다.

# 각 이미지를 가지고 와서 RGB형태로 변환해준 뒤, resize (크기 조정) 해준다.
# 그 값을 numpy배열로 바꾸고, 배열에 추가(append)한다.
# 동시에 category 값도 넣어준다.(Y)
# Y는 0 아니면, 1이니까 label값으로 넣는다.

        X.append(data)  # 숫자 데이터들을 데이터셋으로 만들어준다.
        Y.append(label) 

        if i % 700 == 0:  # 700의 나머지가 0이 됐을때, 출력해서 보여주는것
            print(cat, ':', f)

x = np.array(X)
y = np.array(Y)


# enumerate : 열거하다
# 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능을 가집니다.
# 이 함수는 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 
# 인덱스 값을 포함하는 enumerate 객체를 리턴합니다.
# 보통 enumerate 함수는 for문과 함께 자주 사용됩니다.

# dog_open : 50개

# dog_closed : 50개


# numpy로 변환해서 train_test_split로 데이터셋을 나눈다.

print(x.shape) # (100, 64, 64, 3)
print(y.shape) # (100, 2)

### 데이터 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
xy = (x_train, x_test, y_train, y_test)

print(x_train.shape)    # (80, 64, 64, 3)
print(x_test.shape)     # (20, 64, 64, 3)
print(y_train.shape)    # (80, 2)
print(y_test.shape)     # (20, 2)


### 데이터 SAVE

np.save('D:/Study-bit/project_mini/data.npy', xy)
print('ok', len(y))

np.save('./project_mini/data/multi_image_data.npy', xy)

### 데이터 load

xy = np.load('D:/Study-bit/project_mini/data.npy', allow_pickle=True)

print(xy)


x_train = x_train.reshape(80, 64, 64, 3).astype('float32') /255  
x_test = x_test.reshape(20, 64, 64, 3).astype('float32') /255

print(x_train.shape) # (80, 64, 64, 3)
print(x_test.shape)  # (20, 64, 64, 3)
print(y_train.shape) # (80, 2)
print(y_test.shape)  # (20, 2)


### 모델 만들기

model = Sequential()

model.add(Conv2D(60,(4, 4), input_shape = x_train.shape[1:], activation = 'relu', padding = 'same'))
model.add(Dropout(0.8))
model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(50,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.5))

model.add(Conv2D(48,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.3))

model.add(Conv2D(45, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(35, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(20, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(18,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(14,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(20,(3, 3),activation = 'relu', padding = 'same'))
model.add(Flatten())

model.add(Dense(2, activation = 'sigmoid'))

model.summary()


# callbacks
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# earlystopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1 )

# tensorboard
ts_board = TensorBoard(log_dir = 'graph', histogram_freq = 0,
                        write_graph = True, write_images = True)

# modelcheckpotin   # pred하기 위해 저장
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'

ckpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                          save_best_only = True)


#3. 훈련

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
                validation_split = 0.2, verbose = 2,
                callbacks = [es])

#4. 평가, 예측 (evaluate, predict)

print("-- Evaluate --")
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

# 해당 이미지를 predict

print("-- Predict --")
print('loss: ', loss )
print('acc: ', acc)

# loss:  1.0873358249664307
# acc:  0.699999988079071


# 테스트할 이미지를 변환할 소스 (데이터화)

X = [ ]
filenames = [ ]
files = glob.glob(caltech_dir + "/*/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))

    filenames.append(f)
    X.append(data)


# 그래프 (graph)

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

caltech_dir = 'D:/Study-bit/project_mini/img'


image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    filenames.append(f)
    X.append(data)


X = np.array(X)
X = X.astype(float) / 255
# model = load_model('D:\Study-bit\project_mini\img')

prediction = model.predict(X)
# y_pred = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
                                # lambda 함수를 사용해서 소수점 3자리까지 보여지도록
cnt = 0  # 처음 시작

print(prediction)
# print(y_pred)

for i in prediction:  #  y_pred
# for i in for i in  y_pred :  
    if i[0]>i[1]:             # cnt : 파일 순서 : 0
        print("해당" + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "이미지는 눈을 뜬걸로 추정됩니다.")
                                # 폴더명(카테고리)  +   파일 이름
    else : 
        print("해당" + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "이미지는 눈을 감은것으로 추정됩니다.")
    cnt += 1   # 그 다음 파일을 열기 위해서 +1 을 해준다.

# split함수를 사용해서 파일명과 함께 예측값을 출력해준다.


