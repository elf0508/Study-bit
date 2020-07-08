# 회귀

from sklearn.datasets import load_diabetes
import tensorflow as tf


dataset = load_diabetes()

data = dataset.data
target = dataset.target


print(data.shape) # (442, 10)
print(target.shape) # (442, )

tf.set_random_seed(777)

                               
x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None])


w = tf.Variable(tf.zeros([10, 1]), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)  


cost = tf.reduce_mean(tf.square(hypothesis-y))

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
        cost_val, _ = sess.run([cost, train], feed_dict={x : data, y : target})
        if step % 200 == 0:
            print(step, cost_val)

# 실제 실행 되는 곳
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                            feed_dict={x : data, y : target})

    print("\n Hypothesis : ", h, "\n Correct (y) : ", 
            "\n Accuracy : ", a)


