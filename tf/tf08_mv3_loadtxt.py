# Linear

import tensorflow as tf
import numpy as np


tf.set_random_seed(777)

dataset = np.loadtxt('/data/csv/data-01-test-score.csv',
                    delimiter=',', dtype=np.float32)

# 앞에 3개 가지고, 뒤의 1개 예측
x_data = dataset[:, 0:-1]
y_data = dataset[:, [-1]]

####################################################################

x = tf.placeholder(tf.float32, shape = [None, 3])

y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 최종 나가는 값 : Linear
hypothesis = tf.matmul(x, w) + b  # wx + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

                                                  # 1e-5 : 0.00001
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)  # 최소값으로 만들기 위해 cost값을 minimize 한다.

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):                # hypothesis : 
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict = {x : x_data, y : y_data})
                           
    if step % 10 == 0:
        print(step, "cost : ", cost_val, "\n 예측값 : ", hy_val)

sess.close()
