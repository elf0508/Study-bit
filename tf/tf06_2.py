# tf06_01.py 를 카피해서
# lr을 수정해서 연습
# 0.01 --> 0.1 / 0.001 / 1
# epoch가 2000번을 적게 만들어라.

# placeholder를 이용하면 우리가 linear regression을 만들고 학습을 할때,
# 학습데이터를 우리가 원하는 값으로 넣어줄 수 있다.


import tensorflow as tf

tf.set_random_seed(777)


# x와 y 데이터 생성

# x_train = [1,2,3]
# # y_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32,  shape=[None])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


sess = tf.Session()

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


# hypothesis =  W * x + b
hypothesis = x_train * W + b  

# cost = tf.reduce_mean(tf.square(hypothesis - y ))  # cost :  loss함수 (mse)와 같은것
cost = tf.reduce_mean(tf.square(hypothesis - y_train ))  


# train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate = 1).minimize(cost)


with tf.Session() as sess:
# with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())  
# sess.run(tf.compat.v1.global_variables_initializer())

    # for step in range(2001):  
    # for step in range(1001):  
    for step in range(500):  
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],
                            # feed_dict={x : [1,2,3], y : [3,5,7]})
                          feed_dict={x_train : [1,2,3], y_train : [3,5,7]})
    
        if step % 20 == 0:  # 20번 마다 1번씩 실행
            print(step, cost_val, W_val, b_val)

# predict 해보자

    print("예측 : " , sess.run(hypothesis, feed_dict={x_train : [4]}))
    # 2000번 - 예측 :  [9.008553]
    # 1000번 - 예측 :  [9.000002] / [9.829921]
    # 500번 - 예측 :  [9.000006] /  [9.902318]

    print("예측 : " , sess.run(hypothesis, feed_dict={x_train : [5,6]}))
    # 2000번 - 예측 :  [11.013505 13.018457]
    # 1000번 - 예측 :  [11.000003 13.000004] / [12.310587 14.791254]
    # 500번 - 예측 :  [11.00001  13.000012] / [12.436892 14.971464]

    print("예측 : " , sess.run(hypothesis, feed_dict={x_train : [6,7,8]}))
    # 2000번 - 예측 :  [13.018457 15.02341  17.028362] 
    # 1000번 - 예측 :  [13.000004 15.000005 17.000006] / [14.791254 17.27192  19.752586]
    # 500번 - 예측 :  [13.000012 15.000015 17.00002 ] / [14.971464 17.506039 20.040611]



'''


'''




