import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist


# 데이터 입력

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train[0])

# 데이터 전처리 1. 원핫인코딩

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 데이터 전처리2. 정규화

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

print(x_train.shape, x_test.shape)   # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)   # (60000, 10) (10000, 10)

# 변수
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/ batch_size)  # 60000 / 100 = 600

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)  # dropout

# 레이어
# w1 = tf.Variable(tf.random_normal([784, 512]), name = 'weight1')
w1 = tf.get_variable("w1", shape = [784, 512],      # 1번째 히든 레이어
                      initializer = tf.contrib.layers.xavier_initializer())
print("========== w1 =================")
print("w1 : ", w1)  # shape=(784, 512)

print("=========== b1  =====================")
b1 = tf.Variable(tf.random_normal([512]))
print("b1 : ", b1)  # shape=(512, )

print("========= selu  =======================")
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)  
print("L1 : ", L1)   # shape=(?, 512)

print("=========== dropout ===================")
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)
print("L1 : ", L1)   # shape=(?, 512)

###############################################
print("============= w2 ==================")

w2 = tf.get_variable("w2", shape = [512, 512],    # 2번째 히든 레이어
                      initializer = tf.contrib.layers.xavier_initializer())

b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)  
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)  

print("============= w3 ==================")
w3 = tf.get_variable("w3", shape = [512, 512],     # 3번째 히든 레이어
                      initializer = tf.contrib.layers.xavier_initializer())

b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)  
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)

print("============= w4 ==================")
w4 = tf.get_variable("w4", shape = [512, 256],     # 4번째 히든 레이어
                      initializer = tf.contrib.layers.xavier_initializer())

b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)  
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

print("============= w5 ==================")
w5 = tf.get_variable("w5", shape = [256, 10],     # 5번째 히든 레이어
                      initializer = tf.contrib.layers.xavier_initializer())

b5 = tf.Variable(tf.random_normal([10]))

print("============= hypothesis ==================")
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)  # <-- 최종 나가는 것 / output

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):  # 15
    avg_cost = 0

    for i in range(total_batch):   # 1epoch 에 600번을 돌아라--> 총 9000번 훈련

###############     이 부분 구현 할 것   ########################

         start = i  * batch_size    # 0 
         end = start + batch_size   # 100
         
        #  start = i * batch_size    # 100
        #  end = start + batch_size  # 200

        #  start = i * batch_size    # 200
        #  end = start + batch_size  # 300
         
         batch_xs, batch_ys = x_train[start:end], y_train[start:end]
         
        #  batch_xs, batch_ys = x_train[0:100], y_train[0:100]
        #  batch_xs, batch_ys = x_train[100:200], y_train[100:200]
        #  batch_xs, batch_ys = x_train[200:300], y_train[200:300]

        #  batch_xs, batch_ys = x_train[i:batch_size]
        #  batch_xs, batch_ys = x_train[i:batch_size : batch_size + batch_size]


#########################################################################
           
         feed_dict = {x : batch_xs, y : batch_ys, keep_prob : 0.7}
         c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
         avg_cost += c / total_batch

    print('Epoch : ', '%04d' %(epoch + 1),
          'cost =  {:.9f}'.format(avg_cost))

print('훈련 끝!!!')


prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

print('Acc : ', sess.run(accuracy, feed_dict={x : x_test, y : y_test, keep_prob : 1})) # Acc :  0.9003



