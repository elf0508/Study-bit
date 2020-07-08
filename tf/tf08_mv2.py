
import tensorflow as tf

tf.set_random_seed(777)

x1_data = [[73, 51, 65],
           [92, 98, 11],
           [89, 31, 33],
           [99, 33, 100],
           [17, 66, 79]]   # (5, 3)

y_data = [[152],
          [185],
          [180],
          [205],
          [142]]


x = tf.placeholder(tf.float32, shape = [None, 3])

y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


hypothesis = tf.matmul(x, w) + b  # wx + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

                                                  # 1e-5 : 0.00001
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)  # 최소값으로 만들기 위해 cost값을 minimize 한다.

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):                # hypothesis : 
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict = {x : x1_data, y : y_data})
                           
    if step % 10 == 0:
        print(step, "cost : ", cost_val, "\n 예측값 : ", hy_val)

sess.close()
