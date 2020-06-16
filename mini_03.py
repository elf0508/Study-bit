
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

caltech_dir = 'D:/Study-bit/project_mini/img'

categories = ['dog_open', 'dog_closed']

nb_classes = len(categories)

### 가로, 세로, 채널 쉐이프 정의

image_w = 64
image_h = 64
pixels = image_h * image_w * 3

# 사진의 크기를 64*64 크기로 변환 

### 이미지 파일 Data화 (이미지 파일 변환)
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

dog_open : 50개

dog_closed : 50개
'''

print(x.shape) # (200, 100, 100, 3)
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

cv2.waitKey(0)

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

model.add(Conv2D(600,(4, 4), input_shape = x_train.shape[1:], activation = 'relu', padding = 'same'))
model.add(Dropout(0.8))
model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(500,(3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.5))

model.add(Conv2D(480,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.3))

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

'''
loss:  0.6931487321853638
acc:  0.5
acc:  0.6000000238418579
'''

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

### Open CV  ###

from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import tensorflow as tf

# 칼라로
img_color = cv2.imread("D:/Study-bit/project_mini/img", cv2.IMREAD_COLOR)
cv2.namedWindow('Color')
cv2.imshow('Color', img_color)
cv2.waitKey(0)

# 회색으로
img_gray = cv2.cvtColor("D:/Study-bit/project_mini/img", cv2.COLOR_BGR2GRAY)

cv2.imshow("D:/Study-bit/project_mini/img", img_gray)

cv2.waitKey(0)

cv2.imwrite('saved_image.jpg', img_gray)

cv2.destroyAllWindows()

# showImage()
'''
def showImage():
    imgfile = 'D:/Study-bit/project_mini/img'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    # imread함수는 이미지 파일을 읽기 위한 객체를 리턴한다.
    # 이 함수의 첫번째 인자는 읽고자 하는 이미지의 경로 파일
    # 두 번째 인자는 이미지 파일을 읽는 방식을 나타내는 플래그이다.

    # 플래그 종류
    # cv2.IMREAD_COLOR : 컬러 이밎로 로드, 이미지의 투명한 부분은 모두 무시됨, 디폴트 플래그,정수값은 1
    # cv2.IMREAD_GRAYSCALE : 흑백 이미지로 로드, 정수값은 0
    # cv2.IMREAD_UNCHANGED : 알파채널을 포함하여 이미지 그대로 로드, 정수값은 -1

    cv2.imshow('project_mini/img', img)
    # cv2.imshow()에 의해 번환된 이미지 객체 img를 화면에 나타내기 위한 함수
    # 이 함수는 첫번째 인자는 윈도우 타이틀이며, 두번째 인자는 화면에 표시할 이미지 객체이다.
    cv2.waitKey(0)
    # cv2.waitKey()함수는 지정된 시간동안 키보드 입력을 기다리는 함수
    cv2.destroyAllWindows()
    # 생성한 모든 윈도를 제거한다.

showImage()
'''

'''
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
model = load_model('D:/Study-bit/project_mini/img')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

for i in prediction:
    if i >= 0.5: print("해당 " + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "  이미지는 눈을 뜬걸로 추정됩니다.")
    else : print("해당 " + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "  이미지는 눈을 감은것으로 추정됩니다.")
    cnt += 1
'''
