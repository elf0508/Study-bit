# 선형회귀를 이용한 배추 가격 예측

import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

model = tf.global_variables_initializer()

data = read_csv('price data.csv', sep = ',')

xy = np.array(data, dtype = np.float32)

# 4개의 변인을 입력받는다.

x_data = xy[:, 1:-1]

# 가격 값

y_data = xy[:, [-1]]

# 플레이스홀더를 설정한다.

X = tf.placeholder(tf.float32, shape=[None, 4])

Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")

b = tf.Variable(tf.random_normal([1]), name="bias")

# 가설 설정

hypothesis = tf.matmul(X, W) + b


# 비용 함수를 설정.

cost = tf.reduce_mean(tf.square(hypothesis - Y))


# 최적화 함수를 설정.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)

train = optimizer.minimize(cost)


# 세션을 생성.

sess = tf.Session()


# 글로벌 변수를 초기화.

sess.run(tf.global_variables_initializer())

# 학습을 수행.

for step in range(100001):

    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 500 == 0:

        print("#", step, " 손실 비용: ", cost_)

        print("- 배추 가격: ", hypo_[0])



# 학습된 모델을 저장.

saver = tf.train.Saver()

save_path = saver.save(sess, "./saved.cpkt")

print('학습된 모델을 저장했습니다.')



# 플레이스홀더를 설정한다.

X = tf.placeholder(tf.float32, shape=[None, 4])

Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")

b = tf.Variable(tf.random_normal([1]), name="bias")

# 가설 설정

hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 입력 받습니다.
saver = tf.train.Saver()
model = tf.global_variables_initialier()

# 4가지 변수를 입력 받습니다.
avg_temp = float(input('평균 온도 : '))
min_temp = float(input('최저 온도 : '))
max_temp = float(input('최고 온도 : '))
rain_fall = float(input('강수량 : ' ))

with tf.Session() as sess:
    sess.run(model)

    # 저장된 학습 모델을 파일로부터 불러온다.
    save_path = "./saved.cpkt"
    saver.restore(sess, save_path)

    # 사용자의 입력 값을 이용해 배열을 만든다.
    data = ((avg_temp, min_temp, max_temp, rain_fall),)
    arr = np.arr(data, dtype = np.float32)

    # 예측을 수행한 뒤에 그 결과를 출력
    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict = {X : x_data})

    print(dict[0])

    













