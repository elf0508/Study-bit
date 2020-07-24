from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
from pathlib import Path

import urllib.request
with urllib.request.urlopen('http://python.org/') as response:
   html = response.read()
from urllib.parse import urlparse

baseUrl_1 =  'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' 
# baseUrl_1 = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=image&query=%EC%9D%B4%EC%8A%B9%EA%B8%B0&oquery=&tqi=UxoXhsp0J1ZssSmoKNlssssssmG-136391'   #  이승기
# baseUrl_2 = ' https://search.naver.com/search.naver?where=image&query=%EC%95%84%EC%9D%B4%EC%9C%A0&ie=utf8&sm=tab_she&qdt=0'                 # 아이유

people = {'Celebrity' : ['이승기', '아이유']}

Path("./img_1").mkdir(parents=True, exist_ok=True)
# Path("./img_2").mkdir(parents=True, exist_ok=True)


for k, v in people.items():
    Path("./img_1/" + k).mkdir(parents=True, exist_ok=True)
    for person in v:
        # url = baseUrl_1 + urllib.parse.quote(person)
        url = baseUrl_1 + quote_plus(person)
        html = urlopen(url)
        soup = bs(html, "html.parser")
        # img = soup.findAll(class_='_img_1', limit=50)
        img = soup.findAll(class_='_img_1', limit=100)
        Path("./img_1/" + k + '/' + person).mkdir(parents=True, exist_ok=True)
        n = 1
        for i in img:
            imgUrl = i['data-source']
            with urlopen(imgUrl) as f:
                with open('./img_1' + k + '/' + person + '' + str(n) + '.jpg', 'wb') as h:
                    img = f.read()
                    h.write(img)
            n + 1

print('다운로드 완료')

###############################

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

### 이미지 파일 불러오기 및 카테고리 정의


caltech_dir = 'D:/Study-bit/img_1/Celebrity'

Celebritys = ['아이유', '이승기']

nb_classes = len(Celebritys)

### 가로, 세로, 채널 쉐이프 정의
image_w = 64
image_h = 64
pixels = image_h * image_w * 3

# 사진의 크기를 64*64 크기로 변환

### 이미지 파일 Data화 (이미지 파일 변환)
X = []
Y = []

for idx, cat in enumerate(Celebritys):
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

# 각 이미지를 가지고 와서 RGB형태로 변환해준 뒤, resize 해준다.
# 그 값을 numpy배열로 바꾸고, 배열에 추가(append)한다.
# 동시에 category 값도 넣어준다.(Y)
# Y는 0 아니면, 1이니까 label값으로 넣는다.

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

아이유 : 50개

이승기 : 50개

'''

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

np.save('D:/Study-bit/img_1/Celebrity/data.npy', xy)
print('ok', len(y))

cv2.waitKey(0)
# X.append(img/256)
# Y.append(label)

np.save('D:/Study-bit/img_1/Celebrity/data.npy', xy)
# np.save('D:/Study-bit/img_1/Celebrity/data/multi_image_data.npy', xy)
print('ok', len(y))

### 데이터 load

# X_train, X_test, Y_train, Y_test = np.load('D:/Study-bit/project_mini/data.npy')
xy = np.load('D:/Study-bit/img_1/Celebrity/data.npy', allow_pickle=True)

print(xy)


x_train = x_train.reshape(80, 64, 64, 3).astype('float32') /255  
x_test = x_test.reshape(20, 64, 64, 3).astype('float32') /255

print(x_train.shape) # (80, 64, 64, 3)
print(x_test.shape)  # (20, 64, 64, 3)
print(y_train.shape) # (80, 2)
print(y_test.shape)  # (20, 2)


### 모델 만들기

model = Sequential()

model.add(Conv2D(600,(4, 4), input_shape = x_train.shape[1:], activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(500,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(480,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(450, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(350, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(200, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(180,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(140,(2, 2), activation = 'relu', padding = 'same'))
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

# modelcheckpotin
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


# 테스트할 이미지를 변환할 소스
X = [ ]
filenames = [ ]
files = glob.glob(caltech_dir + "/*/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))

    filenames.append(f)
    X.append(data)

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

#######   추가   #################
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

##########################

from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
# from keras.models import load_model, set_random_seed
import tensorflow as tf
# tensorflow import set_random_seed

# seed = 2
# tf.set_random_seed(seed)
# np.random.seed(seed)

caltech_dir = 'D:/Study-bit/img_1/Celebrity'

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
        print("해당" + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "이미지는 아이유로 추정됩니다.")
                                # 폴더명(카테고리)  +   파일 이름
    else : 
        print("해당" + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "이미지는 이승기로 추정됩니다.")
    cnt += 1   # 그 다음 파일을 열기 위해서 +1 을 해준다.

# split함수를 사용해서 파일명과 함께 예측값을 출력해준다.
