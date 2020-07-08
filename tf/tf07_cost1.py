import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.]
y = [3.,5.,7.]
# y = [1.,2.,3.]

# 세로 : cost  /  가로 : w (1일때, 최소값), 최적의 값

w = tf.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis - y))

w_history = []
cost_history = []

with tf.Session() as sess:
    for i in range(-30, 50):
        curr_w = i * 0.1   # 그림을 그릴 간격
        curr_cost = sess.run(cost, feed_dict={w : curr_w})  # 출력 값

        w_history.append(curr_w)   # w_history의 변동되는 값
        cost_history.append(curr_cost)  # curr_cost의 변동되는 값

plt.plot(w_history, cost_history)
plt.show()



