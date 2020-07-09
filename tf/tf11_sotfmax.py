import tensorflow as tf
import numpy as np

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

with tf.Session():
    print(tf.shape(x_data).eval())   # [8 4]


y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

with tf.Session():
    print(tf.shape(y_data).eval())   # [8 3]


x= tf.placeholder(tf.float32, shape = [None, 4])
y = tf.placeholder(tf.float32, shape = [None, 3])


w = tf.Variable(tf.random_normal([4, 3]), name = 'weight')

# b = tf.Variable(tf.random_normal([3]), name = 'bias')
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')


# 원래 y값과 비교
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)  # wx + b

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x : x_data, y : y_data})

        if step % 200 == 0:
            print(step, cost_val)   # cost_val : loss

# 최적의 w 와 b 가 구해져있다.

########################################

# 예측값 구하기 : hypothesis ( = predict)
    print("=======================================")
    a = sess.run(hypothesis, feed_dict={x : [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print("=======================================")

    b = sess.run(hypothesis, feed_dict={x : [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print("=======================================")

    c = sess.run(hypothesis, feed_dict={x : [[11, 33, 4, 13]]})
    print(c, sess.run(tf.argmax(c, 1)))

    
    # a, b, c를 넣어서 완성할 것 / 힌트 : feed_dict를 수정하라
    print("=======================================")

    all = sess.run(hypothesis, feed_dict={x : [[1, 11, 7, 9],[1, 3, 4, 3], [11, 33, 4, 13]]})
    # feed_dict={x: [np.append(a, 0), np.append(b, 0), np.append(c, 0)]})
    # all = sess.run(hypothesis, feed_dict={x : [[a, b, c]]})
    print(all, sess.run(tf.argmax(all, 1)))


    








