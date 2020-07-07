# 관련 설명 : https://forensics.tistory.com/7

# 관련 설명(동영상 존재) : https://brunch.co.kr/@gnugeun/23
 
  
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

# tf.random_normal([ 숫자 ]) : 0~1 사이의 정규확률분포 값을 생성해주는 함수 / 원하는 shape 대로 만들어줌

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())   #  변수는 항상 초기화를 해야한다.
# print(sess.run(W))  # [2.2086694]


# 가설(모델) 제작 : 아마도 1차원의 분포를 따를 것이다.
# 가설(hypothesis) : 주어진 에 대해서 예측()을 어떻게 할 것인가?

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) 
# cost :  loss함수 (mse)와 같은것
# cost(loss) 함수 : 예측을 얼마나 잘 했는가? 를 표현한 것으로 차이의 제곱에 대한 평균

# 최소의 loss가 최적의 W 값이다.
# 학습을 한다는것 : W 와 b 를 조절해서 cost(loss) 함수의 가장 작은 값을 찾아내는 것

# mse = loss = 전체제곱의 평균 = (y의 예측값 - y값)제곱을 평균것
# 예측에서 실제 y값을 빼면 = 오차
# 오차를 제곱해서 평균을 구하면 2차원의 함수가 제작된다. 
# 제곱하는 이유는 절대값을 사용할 수도 있지만 음수를 방지하기 위해 사용하는 것 같다.

# reduce_mean() : 특정 차원을 제거하고 평균을 구하는 메소드

# square() : 제곱을 구하는 메소드

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# GradientDescentOptimizer : 경사를 따라 내려가는 알고리즘, 기울기를 생각하면 된다.
# Gradient : 경사 / Descent : 내려감 / Optimizer : 손실 함수를 최소화하기 위해 조금씩 variable들을 변경한다.

# 경사 하강법을 통하여 cost를 최소화 시킨다. 하강하는 범위 = learning_rate

# .minimize(cost) : cost 값의 최소값을 찾아주는 함수
# 그러나, 정확하게는 gradient descent 알고리듬에서 gradients를 계산해서 
# 변수에 적용하는 일을 동시에 하는 함수다. 
# W와 b를 적절하게 계산해서 변경하는 역할을 하는데, 그 진행 방향이 cost가 작아지는 쪽이라는 뜻이다.

#우선 with를 통하여 구역을 제정한다.

# Session()의 객체 sess를 사용하여 실행시킨다.

# 그래프 내의 모든 변수의 초기화 연산을 한꺼번에 수행할 때 사용하는 함수
# 우선 tf.global_variables_initializer()을 통하여 변수를 초기화 시킨다.

# sess.run()메소드를 사용하여 학습을 시킨다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 모델의 모든 변수를 초기화한다.
     
    for step in range(2001):  # 2000번을 돌려라
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])
    
    # train은 하지만, 결과값은 출력 X, cost는 cost_val , W는 W_val, b는 b_val 로 출력해라
    # train : 케라스에서 컴파일의 옵티마이져
    
        if step % 20 == 0:  # 20번 마다 1번씩 실행
            print(step, cost_val, W_val, b_val)


'''
import tensorflow as tf
tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [3, 5, 7]
                                                        # 우리가 사용하는 변수와 동일
W = tf.Variable(tf.random_normal([1]), name = 'weight') # 단, Variable사용시 초기화 필수
b = tf.Variable(tf.random_normal([1]), name = 'bias')
                        #_normalization

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화
# print(sess.run(W))                          # [2.2086694]

hypothesis = x_train * W + b                  # model

cost = tf.reduce_mean(tf.square(hypothesis - y_train))   # cost = loss
                                                         # mse

train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost) # cost값 최소화
        # cost를 최소화하기 위해 각 Variable을 천천히 변경하는 optimizer 

with tf.Session() as sess:                        # with을 쓰면 open, close를 안써도 됌 / Session을 계속 사용하기 위해 열어둔다
    sess.run(tf.global_variables_initializer())   # 이 이후로 모든 변수들 초기화
                                     
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b]) # session을 이용해 train 훈련

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

'''




