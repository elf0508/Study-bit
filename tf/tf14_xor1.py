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


# x, y, w, b, hypothesis, cost, train(optimizer)
# 이진분류 / sigmoid 사용
# predict / accuracy 준비해둘것


x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


hypothesis = tf.sigmoid(tf.matmul(x, w) + b)  


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
