# 122를 copy해서 123에 paste
# embedding 없이 LSTM으로 모델 완성

import keras
import numpy as np
from keras.preprocessing.text import Tokenizer

# 영화평 리스트 생성
# docs = ['너무 재밌어요', '최고에요', '참 잘 만든 영화에요',
#         '추천하고 싶은 영화입니다', '한번 더 보고 싶네요',
#         '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
#         '재미없어요', '너무 재미없다', '참 재밌네요']

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정표현 1, 부정표현 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])             # 모델의 최종 출력

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)        # fit_on_texts ; 단어 나열 인덱싱 형태로 반환
print(token.word_index)

# 단어의 수치화 ; texts_to_sequences()
x = token.texts_to_sequences(docs)
print(x)


# 원핫인코딩
pad_x = keras.preprocessing.sequence.pad_sequences(x, padding = 'pre')
print(pad_x)            # (12, 5)

pad_x = pad_x.reshape(12, 5, 1)   # <-- 3차원으로 shape를 맞춰줘야한다.
print(pad_x)            # (12, 5, 1)

word_size = len(token.word_index) + 1
print(f'전체 토큰 사이즈 : {word_size}')     # 전체 토큰 사이즈 : 25


## 모델링
model = keras.models.Sequential()
model.add(keras.layers.LSTM(10, input_shape = (5, 1),
                            return_sequences = True))
model.add(keras.layers.LSTM(3))
# model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))

model.summary()


## 컴파일
EPOCH = 30
model.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
              loss = keras.losses.binary_crossentropy,
              metrics = ['accuracy'])
model.fit(pad_x, labels, epochs = EPOCH)

acc = model.evaluate(pad_x, labels)[1]
print(f'Accuracy : {acc}')         

'''
from keras.preprocessing.text import Tokenizer
import numpy as np 

# 122를 그대로 카피해서 123 완성
# embedding을 빼고 LSTM으로 구성

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요',           # x
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요','너무 재미없다', '참 재밋네요']
    
# 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])              # y

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)


x = token.texts_to_sequences(docs)   
print(x)                            


from keras.preprocessing.sequence import pad_sequences    
pad_x = pad_sequences(x, padding = 'post', value = 0.0)   
print(pad_x)
print(pad_x.shape)                                        # (12, 5)

pad_x = pad_x.reshape(12, 5, 1)                                             



word_size = len(token.word_index) + 1 # [0] 포함
print('전체 토큰 사이즈 :', word_size)                      # 전체 토큰 사이즈 : 25


#2. model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
# model.add(Embedding(word_size, 10, input_length = 5))    
# model.add(Embedding(20, 10, input_length = 5))               
# model.add(Embedding(25, 10))                              
model.add(LSTM(3, input_shape = (5, 1)))                                        
# model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

#3. compile, fit
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)                           

#################################################

from keras.preprocessing.text import Tokenizer
import numpy as np 

# 122를 그대로 카피해서 123 완성
# embedding을 빼고 LSTM으로 구성

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요',           # x
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요','너무 재미없다', '참 재밋네요']
    
# 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])              # y

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)


x = token.texts_to_sequences(docs)   
print(x)                            


from keras.preprocessing.sequence import pad_sequences    
pad_x = pad_sequences(x, padding = 'post', value = 0.0)   
print(pad_x)
print(pad_x.shape)                                        # (12, 5)

pad_x = pad_x.reshape(12, 5, 1)                                             



word_size = len(token.word_index) + 1 # [0] 포함
print('전체 토큰 사이즈 :', word_size)                      # 전체 토큰 사이즈 : 25


#2. model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
# model.add(Embedding(word_size, 10, input_length = 5))    
# model.add(Embedding(20, 10, input_length = 5))               
# model.add(Embedding(25, 10))                              
model.add(LSTM(3, input_shape = (5, 1)))                                        
# model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

#3. compile, fit
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)                                   # acc : 0.5  

'''