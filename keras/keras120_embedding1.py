# Tokenizer : 낱말 분석

from keras.preprocessing.text import Tokenizer

text = "나는 맜있는 밥을 먹었다" 

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)

# {'나는': 1, '맜있는': 2, '밥을': 3, '먹었다': 4}  <-- fit_on_texts
# 수치화 : 한개의 문장을 단어 단위로 잘랐다(조사 포함). <-- word_index

x = token.texts_to_sequences([text])
print(x)

# [[1, 2, 3, 4]]
# 문자를 순서대로 수치화  <-- texts_to_sequences

# 원핫 인코딩
from keras.utils import to_categorical

word_size = len(token.word_index) + 1
# to_categorical 이 0부터 시작해서 + 1 을 해줬다.

x = to_categorical(x, num_classes = word_size)

print(x)

# [[[0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 1.]]]   수치화 해서, 4바이 5로 나왔다.


