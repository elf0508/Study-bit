# 데이터셋 :  Naver sentiment movie corpus를 사용.

# 이 데이터셋은 네이버 영화의 리뷰 중 영화당 100개의 리뷰를 모아 총 200,000개의 리뷰(train: 15만, test: 5만)로 이루어져있고, 
# 1~10점까지의 평점 중에서 중립적인 평점(5~8점)은 제외하고 1~4점을 긍정으로, 9~10점을 부정으로 동일한 비율로 데이터에 포함.
# 데이터는 id, document, label 세 개의 열로 이루어져있다. 
# id는 리뷰의 고유한 key 값이고, document는 리뷰의 내용, label은 긍정(0)인지 부정(1)인지를 나타낸다. 
# txt로 저장된 데이터를 처리하기 알맞게 list 형식으로 받아서 사용.

# 데이터 불러오기

from keras.datasets import imdb
# import nltk


# 학습 데이터와 테스트 데이터로 분리 (빈도가 높은 10000개 단어 대상)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 결과를 구해 출력하는 시간이 다소 걸림....
print(train_data[0])            # [1, 14, 22, ..., 178, 32]
print(train_labels[0])          # 1

# 가장 높은 인덱스(9999)를 구하여 출력하기
print( max([max(sequence) for sequence in train_data]) )  # 9999

#-----------------------------------------------------------------
# 리뷰 데이터에서 하나씩 원래 단어로 바꿔서 출력하기

# 단어 인덱스 생성 
word_index = imdb.get_word_index()   

# 인덱스와 단어로 매핑한 새로운 사전 생성
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# 리뷰 디코딩 하여 출력하기
# 인덱스 0,1,2는 패딩, 문서시작, 사전에 없음을 위한 인덱스라서 3 건너 뛴다.
# 찾는 인덱스가 없으면 '?' 문자열을 반환
decode_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decode_review)


#----------------------------------------------------------
# 리스트 데이터를 텐서로 바꾸기
# 정수 리스트 신경망에 넣을 수 없으므로 값을 0과 1 벡터로 변환
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 크기 len(sequence), dimenton인 행렬 생성 (모든 값은 0)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0      # 인덱스 i 위치를 1.0 설정
    return results


# 훈련용 데이터를 벡터로 변환
x_train = vectorize_sequences(train_data)

# 훈련용 데이터를 벡터로 변환
x_test = vectorize_sequences(test_data)


print(x_train[0])           # 출력 테스트  [0. 1. 1. ... 0. 0. 0.]


# 레이블을 벡터로 변경
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# 출력 테스트
print(y_train)  # [1. 0. 0. ... 0. 1. 0.]
print(y_test)   # [0. 1. 1. ... 0. 0. 0.]
