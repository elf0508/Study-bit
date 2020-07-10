# DNN

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist


# 데이터 입력

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])

# 데이터 전처리 1. 원핫인코딩

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(y_train[0])  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# print(y_test[0])   # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]


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

print("============= hypothesis : (예측) 가정 수립 ==================")
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)  # <-- 최종 나가는 것 / output

#------------------------------- compile --------------------------------

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

#-------------------------------- fit ------------------------------------
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


'''
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist          
import numpy as np               
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28*28).astype('float32')/255.
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.

print(x_train.shape)   # (60000, 784)
print(y_train.shape)   # (60000, 10) 

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)  # 60000 / 100

x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])

keep_prob = tf.placeholder(tf.float32)        # dropout


                                # input / output   
# w = tf.variable(tf.random_normal([784, 512]), name = 'weight1')  # 동일  
w1 = tf.get_variable('w1', shape=[784, 512],                        # 초기 변수가 없으면 알아서 할당함/ 파라미터 많음
                    initializer=tf.contrib.layers.xavier_initializer())
print('=========== w1 ===========')                                            
print(w1)                           # shape=(784, 512) 
b1 = tf.Variable(tf.random_normal([512]), name = 'bias')
print('=========== b1 ===========')
print(b1)                           # shape=(512,) 
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)
print('=========== L1 ===========')
print(L1)                           # shape=(?, 512)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)   
print('=========== L1 ===========')
print(L1)                           # shape=(?, 512)

w2 = tf.get_variable('w2', shape=[512, 512],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b2 = tf.Variable(tf.random_normal([512]), name = 'bias')
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

w3 = tf.get_variable('w3', shape=[512, 512],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b3 = tf.Variable(tf.random_normal([512]), name = 'bias')
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

w4 = tf.get_variable('w4', shape=[512, 256],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b4 = tf.Variable(tf.random_normal([256]), name = 'bias')
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

w5 = tf.get_variable('w5', shape=[256, 10],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b5 = tf.Variable(tf.random_normal([10]), name = 'bias')
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)                           # 마지막 output / dropout필요 없음


loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):            # 15
    ave_cost = 0

    for i in range(total_batch):                # 600
        start = i*batch_size                    # 실질적으로 데이터 100개씩을 가지고 15 * 600번 돌려서 학습이 된다.
        end = start + batch_size                # 조금식 선을 그린다 / 배치 사이즈를 안쓰면 한번에(60000개로) 대충 그리는 거

        batch_xs, batch_ys = x_train[start : end], y_train[start : end]
        
        feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.9}           # (1 - keep_prob)만큼 dropout한다.
        c, _=sess.run([loss, optimizer], feed_dict=feed_dict) 
        ave_cost += c / total_batch

    print('Epoch :', '%04d'%(epoch+1),
          'loss =', '{:.9f}'.format(ave_cost))

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc :', sess.run(accuracy, feed_dict = {x:x_test, y:y_test, keep_prob:1})) 
# keras에서 evaluate나 predict 할 때에는 가중치는 적용이 되지만 / 드랍아웃이 적용되어 있지 않는다.
# 통상적으로 keep_prob는 1이 들어간다.

'''
