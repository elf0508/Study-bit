from sklearn.datasets import load_diabetes
import tensorflow as tf

diabetes = load_diabetes()

x_data = diabetes.data
y_data = diabetes.target

y_data = y_data.reshape(-1, 1)

print(x_data.shape)           # (442, 10)
print(y_data.shape)           # (442,)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name = 'weight')
                                # x의 열의 값과 동일 해야 함(행렬 계산) 

b = tf.Variable(tf.random_normal([1]), name = 'bias') 

hypothesis = tf.matmul(x, w) + b                      # wx + b (행렬 곱)

cost =  tf.reduce_mean(tf.square( hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-6) # 0.00001

train = optimizer.minimize(cost)

# 준비 상태
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),
                            dtype = tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _= sess.run([cost, hypothesis, train],
                                feed_dict = {x: x_data, y: y_data})

    if step % 20 == 0 :
        print(step, 'cost :',cost_val, '\n 예측값 :', hy_val)

# 실제 실행 되는 곳
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                            feed_dict={x : x_data, y : y_data})

    print("\n Hypothesis : ", h, "\n Correct (y) : ", c,
            "\n Accuracy : ", a)