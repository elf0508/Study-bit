# sigmoid

import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1, 2],
           [2, 3],
           [3, 1],
           [4, 3],
           [5, 3],
           [6, 2]]   

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]


x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


hypothesis = tf.sigmoid(tf.matmul(x, w) + b)  


# cost = tf.reduce_mean(tf.square(hypothesis - y))

# 시그모이드를 직접 구현한 함수
# sigmoid = tf.compat.v1.div(1., 1. + tf.compat.v1.exp(tf.matmul(x, w)))


# 2. Cost function 최소화

#cost function(logistic regression에서, W와 b를 찾기 위한 cost)

# 시그모이드에서 앞에 마이너스가 붙는 이유는 음수가 안나오게 하기 위해서
# 로그안에 들어가는 x값이 sigmoid를 거치기때문에
# 무조건 0~1사이라서 모든 값이 음수가 나온다.
# 음수 싫으면 출력할때 -cost 이렇게 출력한다.

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

    print("\n Hypothesis : ", h, "\n Correct (y) : ", 
            "\n Accuracy : ", a)

