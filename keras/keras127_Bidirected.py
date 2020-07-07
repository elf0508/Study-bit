# reuters : 뉴스 기사로 카테고리를 나누는 것 / 46개의 카테고리
# 1만개 --> 8천개로 훈련 --> 카테고리별로 나누는 것

from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000) #, test_split = 0.2)

print(x_train.shape, x_test.shape)  # (8982, ) (2246, )
print(y_train.shape, y_test.shape)  # 8982, ) (2246, )

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))   # 87   크기가 일정하지 않다. --> 빈 자리는 0으로 채운다.

# y의 카테고리 개수 출력
category = (np.max(y_train)) + 1
print("카테고리 : ", category)   #  카테고리 :  46  / 0 ~ 45까지 있다. /다중분류

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# 판다스 / groupby
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)


from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 111, padding = 'pre') 
x_test = pad_sequences(x_test, maxlen = 111, padding = 'pre') 
# padding = 'pre' --> 앞에서부터 0으로 13개 채운다.

# print(len(x_train[0]))
# print(len(x_train[-1]))

# 원핫인코딩

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)   

# 2.모델

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten 
from keras.layers import Conv1D, Dropout, Activation, MaxPooling1D, Bidirectional
model = Sequential()

# model.add(Embedding(1000, 128, input_length = 111))
model.add(Embedding(2000, 128))
# model.add(Embedding(10000, 128))

model.add(Conv1D(10, 5, padding = 'valid', activation = 'relu', strides = 1))

model.add(MaxPooling1D(pool_size = 4))

model.add(Bidirectional(LSTM(10)))   # Bidirectional 은 LSTM 랩핑
# model.add(LSTM(10))

# model.add(LSTM(100))

model.add(Dense(1, activation= 'sigmoid'))


model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                    metrics = ['acc'])

history = model.fit(x_train, y_train, batch_size = 100, epochs = 10,
                        validation_split = 0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)

# 그래프
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker ='.', c ='red', label ='TestSet Loss')
plt.plot(y_loss, marker ='.', c ='blue', label ='TrainSet Loss')
plt.legend(loc ='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()

# acc :  0.8227599859237671

from keras.preprocessing.text import Tokenizer
token = Tokenizer()
token.fit_on_texts(imdb.get_word_index())
token.sequences_to_texts(x_train[0:1])

# 1. imdb 검색해서 데이터 내용 확인
# 훈련용 리뷰는 25,000개, 테스트용 리뷰는 25,000개, 카테고리는 2개

# 2. word_size 전체데이터 부분 변경해서 제일 좋은 값 확인

# 3. groupby() : 같은 값을 하나로 묶어 통계 또는 집계 결과를 얻기 위해 사용하는 것

# 예를 들어 도시(city) 별로 가격(price) 평균을 구하고 싶은 경우 
# 평균값을 구해주는 메서드로 mean을 사용한다.

# df.groupby('city').mean()

# 그룹 지정은 여러 개를 지정할 수도 있다.
# 도시(city)와 과일(fruits)로 평균을 구한다면,

# df.groupby(['city', 'fruits']).mean()

# 도시별로 그룹화하고 다시 과일 종류별로 그룹이 된 평균값을 얻을 수 있다.
# groupby를 사용하면 기본으로 그룹 라벨이 index가 된다.
# index를 사용하고 싶은 않은 경우에는 as_index=False 를 설정하면 된다.

# df.groupby(['city', 'fruits'], as_index=False).mean()

################################################

# 인덱스를 단어로 바꿔주는 함수
# 정수를 단어로 다시 변경하기

# # 정수와 문자열을 매핑한 딕셔너리 객체에 질의하는 헬퍼를 만듬

# # 단어와 정수 인덱스를 매핑한 딕셔너리

# word_to_index = imdb.get_word_index()
# index_to_word={}
# for key, value in word_to_index.items():
#     index_to_word[value] = key

# print('빈도수 상위 1번 단어: {}'.format(index_to_word[1]))
# print('빈도수 상위 3941번 단어: {}'.format(index_to_word[3941])

'''
from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt       

#1. data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000) # train:test = 50:50

print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)


# print(x_train[0])
# print(y_train[0])        # 1


print(len(x_train[0]))   # 218 

# x_train내용 보기
word_to_index = imdb.get_word_index() # {'단어': index, }
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key        # {'index' : 단어}
print(' '.join([index_to_word[index] for index in x_train[0]]))
                            

# y의 카테고리 개수 출력
category = np.max(y_train) + 1     # index가 0부터 시작함으로 + 1 해줌
print('카테고리 :', category)       # 카테고리 : 2

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
# print(y_bunpo)                   # [0 1]


y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
# print(bbb)                       # 0    12500   : 부정
                                   # 1    12500   : 긍정
# print(bbb.shape)                 # (2,)



from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 111, padding = 'pre')
x_test = pad_sequences(x_test, maxlen = 111, padding = 'pre')


# print(len(x_train[0]))            # 111
# print(len(x_train[-1]))           # 111

# y_train = to_categorical(y_train)  # 이진 분류임으로 원핫인코딩 필요없음
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)   # (25000, 111) (25000, 111)


#2. model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPooling1D

model = Sequential()
# model.add(Embedding(1000, 128, input_length = 111))
model.add(Embedding(2000, 128))

model.add(Conv1D(32, 5, padding = 'valid', activation = 'relu', strides = 1))
model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(32, return_sequences= True))
model.add(LSTM(32))
model.add(Dense(10))
model.add(Dense(1, activation = 'sigmoid'))

# model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
               metrics = ['acc'])

history = model.fit(x_train, y_train, batch_size= 128, epochs = 10,
                     validation_split = 0.1)

acc = model.evaluate(x_test, y_test)[1]
print('acc :', acc)                          # acc : 0.8343600034713745             

# 그림을 그리자
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'blue', label = 'TraintSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# 1. imdb 검색해서 데이터 내용 확인
# 2. word_size 전체 데이터 부분 변경해서 최상값 확인
# 3. 주간과제 : groupby() 사용법 숙지할 것
# 4. 인덱스를 단어로 바꿔주는 함수 찾을 것 : .index_to_word[]

'''


