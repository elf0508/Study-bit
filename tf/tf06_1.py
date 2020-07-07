import tensorflow as tf

tf.set_random_seed(777)

# 텐서 플로우를 import하고 난수 발생 seed를 설정
# random_seed(777)하는 과정은 컴퓨터 마다 난수발생하는 방법을 통일 시키는 것 같다.

# x와 y 데이터 생성

x_train = [1,2,3]
# y_train = [1,2,3]
y_train = [3,5,7]



# 가중치와 바이어스를 설정
# variable를 만드는데 값을 random_normal 정규화 분포의 값으로 무작위로 제작, 
# 모양은 스칼라 이기 때문에 1

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())   #  변수는 항상 초기화를 해야한다.
# print(sess.run(W))  # [2.2086694]


# 가설(모델) 제작 : 아마도 1차원의 분포를 따를 것이다.

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # cost :  loss함수 (mse)와 같은것

# 최소의 loss가 최적의 W 값이다.
# mse = loss = 잔차제곱의 평균 = (y의 예측값 - y값)제곱을 평균것
# 예측에서 실제 y값을 빼면 = 오차
# 오차를 제곱해서 평균을 구하면 2차원의 함수가 제작된다. 
# 제곱하는 이유는 절대값을 사용할 수도 있지만 음수를 방지하기 위해 사용하는 것 같다.

# reduce_mean() : 평균 구하는 메소드

# square() : 제곱을 구하는 메소드

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
# 경사 하강법을 통하여 cost를 최소화 시킨다. 하강하는 범위 = learning_rate

#우선 with를 통하여 구역을 제정한다.

# Session()의 객체 sess를 사용하여 실행시킨다.

# 우선 tf.global_variables_initializer()을 통하여 변수를 초기화 시킨다.

# sess.run()메소드를 사용하여 학습을 시킨다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):  # 2000번을 돌려라
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])
    
    # train은 하지만, 결과값은 출력 X, cost는 cost_val , W는 W_val, b는 b_val 로 출력해라
    # train : 케라스에서 컴파일의 옵티마이져
    
        if step % 20 == 0:  # 20번 마다 1번씩 실행
            print(step, cost_val, W_val, b_val)


'''
import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
y = [3, 5, 7]

x_train = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)
                                                        
W = tf.Variable(tf.random_normal([1]), name = 'weight') 
b = tf.Variable(tf.random_normal([1]), name = 'bias')
                        #_normalization

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))                          

hypothesis = x_train * W + b                  

cost = tf.reduce_mean(tf.square(hypothesis - y_train))   

train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost) 
         

with tf.Session() as sess:                        
    sess.run(tf.global_variables_initializer())   
                                     
    for step in range(2001):
        # _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:x, y_train:y}) 


        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

'''




