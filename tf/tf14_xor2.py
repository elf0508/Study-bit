# 이진분류 / sigmoid 사용

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = np.float32)

with tf.Session():
    print(tf.shape(x_data).eval()) # [4 2]

y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

with tf.Session():
    print(tf.shape(y_data).eval())  # [4 1]

##################################################


x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

# 레이어 구성                         100: 히든 레이어의 값
w1 = tf.Variable(tf.zeros([2, 100]), name = 'weight1')
b1 = tf.Variable(tf.zeros([100]), name = 'bias')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)  
# if 케라스 
# model.add(Dense(100, input_dim = 2))

w2 = tf.Variable(tf.zeros([100, 50]), name = 'weight1')
b2 = tf.Variable(tf.zeros([50]), name = 'bias')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2) 
# model.add(Dense(50))


w3 = tf.Variable(tf.zeros([50, 1]), name = 'weight1')
b3 = tf.Variable(tf.zeros(1), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)  
# model.add(Dense(1))  <-- output


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * 
                        tf.log(1 - hypothesis))

                                                  # 1e-5 : 0.00001
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)  # 최소값으로 만들기 위해 cost값을 minimize 한다.

# 준비 상태
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),
                            dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x : x_data, y : y_data})
        if step % 200 == 0:
            print(step, cost_val)

# 실제 실행 되는 곳
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                            feed_dict={x : x_data, y : y_data})

    print("\n Hypothesis : ", h, "\n Correct (y) : ", c,
            "\n Accuracy : ", a)
