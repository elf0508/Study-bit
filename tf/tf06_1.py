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
cost = tf.reduce_mean(tf.square(hypothesis - y_train ))  # cost :  loss함수 (mse)와 같은것


train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


with tf.Session() as sess:
# with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())  # initializer() : 변수 선언하겠다.
# global_variables_initializer() : 초기화는 1번만 된것이다.
# sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):  # 2000번을 돌려라
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],
                            # feed_dict={x : [1,2,3], y : [3,5,7]})
                          feed_dict={x_train : [1,2,3], y_train : [3,5,7]})
    
    # train은 하지만, 결과값은 출력 X, cost는 cost_val , W는 W_val, b는 b_val 로 출력해라
    
        if step % 20 == 0:  # 20번 마다 1번씩 실행
            print(step, cost_val, W_val, b_val)

# predict 해보자

    print("예측 : " , sess.run(hypothesis, feed_dict={x_train : [4]}))
    # 예측 :  [9.008553]

    print("예측 : " , sess.run(hypothesis, feed_dict={x_train : [5,6]}))
    # 예측 :  [11.013505 13.018457]

    print("예측 : " , sess.run(hypothesis, feed_dict={x_train : [6,7,8]}))
    # 예측 :  [13.018457 15.02341  17.028362] 




'''
import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
y = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape = [None])    # 현재 shape는 모른다.
y_train = tf.placeholder(tf.float32, shape = [None])
                                                        
W = tf.Variable(tf.random_normal([1]), name = 'weight') # 난수를 주는 이유 
b = tf.Variable(tf.random_normal([1]), name = 'bias')   # : 시작 위치가 달라져도 최적의 값을 찾아가는 것을 보기 위함 
                        #_normalization                 # / 상수를 써도 상관 없다.

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))                          

hypothesis = x_train * W + b                  

cost = tf.reduce_mean(tf.square(hypothesis - y_train))   

train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost) 
         

with tf.Session() as sess:                        
    sess.run(tf.global_variables_initializer())          # 변수에 메모리를 할당하고 초기값을 설정하는 역할
                                     
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 
        # _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:x, y_train:y}) 


        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

'''




