# 분류
# RNN - 수치로 구현해보기 / 3차원

import tensorflow as tf
import numpy as np   

# 1. data =  hihello

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype = np.str).reshape(-1, 1)
# _data = np.array([['h'], ['i'], ['h'], ['e'], ['l'], ['l'], ['o']])

print(_data.shape)  # (7, 1)
print(_data)        
#  [['h']
#   ['i']
#   ['h']
#   ['e']
#   ['l']
#   ['l']
#   ['o']]

print(type(_data))  # <class 'numpy.ndarray'>

# 원핫인코딩 = 문자별로 수치화 가능하다.
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()   # 변환

print("=============================================")

print(_data)         
# [[0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]

print(type(_data))   # <class 'numpy.ndarray'>
print(_data.dtype)   # float64

x_data = _data[:6, ]  # hihell    (1, 6, 5)  30개의 데이터를 5개씩 잘라서 작업중 / LSTM 가능하다.
y_data = _data[1:, ]  #  ihello   (6, 1) / -->  y값이 (1, 6) 으로 나와야한다.

print("============== x ====================")
print(x_data)
# [[0. 1. 0. 0. 0.]     (6, 5)
#  [0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]]
print("=============  y ====================")
print(y_data)
# [[0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
print("=====================================")


# 원핫인코딩 한 걸 원래 수치로 바꾸려면

y_data =np.argmax(y_data, axis = 1)
print("============= y argmax ======================")
print(y_data)         # [2 1 0 3 3 4]  <-- int형
print(y_data.shape)   # (6, )  shape 바꿔야 한다.

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)
print("=====================================")

print(x_data.shape)   # (1, 6, 5)
print(y_data.shape)   # (1, 6)


sequence_length = 6
input_dim = 5
output = 5
batch_size = 1   # 전체 행

# X = tf.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y = tf.placeholder(tf.float32, (None, sequence_length))

X =  tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y =  tf.compat.v1.placeholder(tf.float32, (None, sequence_length))
Y =  tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

print(X)   # shape=(?, 6, 5)
print(Y)   # shape=(?, 6)


# 2. 모델 구성

# model.add(LSTM(output, input_shape = (6, 5)))

# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
                      #     model.add(LSTM)
print("============ hypothesis ==============")

print(hypothesis)    # shape=(?, 6, 5)
print("=====================================")

# 3-1. 컴파일
# weights = tf.ones([1, 6])   # Y의 shape와 같다. 선형을 디퐅트 값으로 잡겠다.
weights = tf.ones([batch_size, sequence_length])   

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights)

cost = tf.reduce_mean(sequence_loss)  # 전체에 대한 평균

# train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.argmax(hypothesis, axis = 2)
print(prediction)

# 3-2. 훈련(fit)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
        result = sess.run(prediction, feed_dict = {X : x_data})
        print(i, "loss : ", loss, "prediction : ", result, "true Y : ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str : ", ''.join(result_str))






