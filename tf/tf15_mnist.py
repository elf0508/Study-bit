# 케라스 2.3.1 설치
# 레이어를10개 연결 해보기


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist          
import numpy as np               
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)   # (60000, 28, 28)
print(y_train.shape)   # (60000,)  

x_train = x_train.reshape(-1, 28*28).astype('float32')/255.
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.


x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])


# 1                           # input / output   
w1 = tf.Variable(tf.random_normal([784, 8]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([8]), name = 'bias1')
layer1 = tf.matmul(x, w1) + b1

# 2                       
w2 = tf.Variable(tf.random_normal([8, 16]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([16]), name = 'bias1')
layer2 = tf.matmul(layer1, w2) + b2

# 3                       
w3 = tf.Variable(tf.random_normal([16, 32]), name = 'weight2')
b3 = tf.Variable(tf.random_normal([32]), name = 'bias1')
layer3 = tf.matmul(layer2, w3) + b3

# 4                       
w4 = tf.Variable(tf.random_normal([32, 64]), name = 'weight2')
b4 = tf.Variable(tf.random_normal([64]), name = 'bias1')
layer4 = tf.matmul(layer3, w4) + b4

# 5                       
w5 = tf.Variable(tf.random_normal([64, 128]), name = 'weight2')
b5 = tf.Variable(tf.random_normal([128]), name = 'bias1')
layer5 = tf.matmul(layer4, w5) + b5

# 6                       
w6 = tf.Variable(tf.zeros([128, 258]), name = 'weight2')
b6 = tf.Variable(tf.zeros([258]), name = 'bias1')
layer6 = tf.matmul(layer5, w6) + b6

# 7                       
w7 = tf.Variable(tf.zeros([258, 128]), name = 'weight2')
b7 = tf.Variable(tf.zeros([128]), name = 'bias1')
layer7 = tf.matmul(layer6, w7) + b7

# 8                       
w8 = tf.Variable(tf.zeros([128, 32]), name = 'weight2')
b8 = tf.Variable(tf.zeros([32]), name = 'bias1')
layer8 = tf.matmul(layer7, w8) + b8

# 9                       
w9 = tf.Variable(tf.zeros([32, 10]), name = 'weight2')
b9 = tf.Variable(tf.zeros([10]), name = 'bias1')
hypothesis = tf.nn.softmax(tf.matmul(layer8, w9) + b9)                 # 마지막 output_layer

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)
# model.add(Dense(1, input_dim = 50))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #3. 
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_test = sess.run(tf.one_hot(y_test, 10))
    

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict = {x:x_train, y:y_train})

        if step % 200 ==0:
            print(step, cost_val)

    # 최적의 W와 b가 구해져 있다
    a = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = sess.run(tf.argmax(a, 1))
    print(a, y_pred )

    # #1. Accuracy - sklearn
    # y_pred = sess.run(tf.one_hot(y_pred, 3))
    # # y_test = sess.run(tf.argmax(y_test, 1))       # y_test(원핫), y_pred(argmax) 의 모양을 같게 만들어 주기 위해서
    # acc = accuracy_score(y_test, y_pred)
    # print('Accuracy :', acc)

    #2. Accuracy - tensorflow
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

'''
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# 탠서플로의 원핫 인코딩
# aaa = tf.one_hot(y, ???)
seed = 0
tf.set_random_seed(seed)

(x_data, y_data), (x_test, y_test) = mnist.load_data()

print(x_data.shape)
print(y_data.shape)


sess = tf.Session()
y_data = tf.one_hot(y_data,depth=10,on_value=1,off_value=0).eval(session=sess)
y_test = tf.one_hot(y_test,depth=10,on_value=1,off_value=0).eval(session=sess)
sess.close()
print(y_data.shape)

x_data = x_data.reshape(-1,x_data.shape[1]*x_data.shape[2])
print(x_data.shape)
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])
print(x_test.shape)

x_col_num = x_data.shape[1] # 4
y_col_num = y_data.shape[1] # 3

print(x_col_num)
print(y_col_num)

x = tf.placeholder(tf.float32, shape=[None, x_col_num])
y = tf.placeholder(tf.float32, shape=[None, y_col_num])

w1 = tf.Variable(tf.zeros([x_col_num, 512]), name = 'weight1') # 다음 레이어에 100개를 전달? 노드의 갯수와 동일하다고 봐도 무방?
b1 = tf.Variable(tf.zeros([512]), name = 'bias1') # 자연스럽게 100을따라감 ?? 왜?
layer1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([512, 256]), name='weight2')
b2 = tf.Variable(tf.random_normal([256]),name = 'bias2')
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.Variable(tf.random_normal([256, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]),name = 'bias3')
h = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h),axis=1)) # loss ... 계산 방법 ...

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(loss) # 어떻게 쓰는지 어떻게 계산하는지 지금은 일단 쓰고 시간이 많을때 꼭!!! 공부하라 경사하강법

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        _, _, cost_val = sess.run([h, opt, loss], feed_dict={ x: x_data, y: y_data})

        # if i % 200 == 0 :
        print( i , cost_val)
    
    pred = sess.run(h, feed_dict={x:x_test}) # keras model.predict(x_test_data)
    pred = sess.run(tf.argmax(pred, 1)) # tf.argmax(a, 1) 안에 값들중에 가장 큰 값의 인덱스를 표시하라
    # pred = pred.reshape(-1,1)
    print(pred)

    y_test = sess.run(tf.argmax(y_test,1))
    print(y_test)

    acc = tf.reduce_mean(tf.compat.v1.cast(tf.equal(pred, y_test),tf.float32))
    acc = sess.run(acc)
    print(acc)

####################################

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)


x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))

y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)

# y_data = y_data.reshape(y_data.shape[0],1)

x = tf.placeholder(tf.float32, shape = [None, 28*28])
y = tf.placeholder(tf.float32, shape = [None, 10])

w = tf.Variable(tf.random_normal([28*28, 100]), name="weight")
b = tf.Variable(tf.random_normal([100]), name="bias")
layer = tf.nn.softmax(tf.matmul(x, w) +b)
#model.add(Dense(100,input_shape=(2,)))

w = tf.Variable(tf.random_normal([100, 50]), name="weight")
b = tf.Variable(tf.random_normal([50]), name="bias")
layer = tf.nn.softmax(tf.matmul(layer, w) +b)
#model.add(Dense(50))

w = tf.Variable(tf.random_normal([50, 50]), name="weight")
b = tf.Variable(tf.random_normal([50]), name="bias")
layer = tf.nn.softmax(tf.matmul(layer, w) +b)
#model.add(Dense(50))


w = tf.Variable(tf.random_normal([50, 50]), name="weight")
b = tf.Variable(tf.random_normal([50]), name="bias")
layer = tf.nn.softmax(tf.matmul(layer, w) +b)
#model.add(Dense(50))


w = tf.Variable(tf.random_normal([50, 50]), name="weight")
b = tf.Variable(tf.random_normal([50]), name="bias")
layer = tf.nn.softmax(tf.matmul(layer, w) +b)
#model.add(Dense(50))


w = tf.Variable(tf.random_normal([50, 50]), name="weight")
b = tf.Variable(tf.random_normal([50]), name="bias")
layer = tf.nn.softmax(tf.matmul(layer, w) +b)
#model.add(Dense(50))


w = tf.Variable(tf.random_normal([50, 50]), name="weight")
b = tf.Variable(tf.random_normal([50]), name="bias")
layer = tf.nn.softmax(tf.matmul(layer, w) +b)
#model.add(Dense(50))

w = tf.Variable(tf.random_normal([50, 10]), name="weight")
b = tf.Variable(tf.random_normal([10]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(layer, w) +b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
# train = optimizer.minimize(cost)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for step in range(300):
        _,loss_val,hypo_val=sess.run([optimizer,loss,hypothesis],feed_dict={x:x_train,y:y_train})
        # if step % 10==1:
        #     print(loss_val)
        print(f"step:{step},loss_val:{loss_val}")
        # 실제로 실현되는 부분
    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
    
    #정확도
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy : ",sess.run(accuracy, feed_dict = {x :x_test, y :y_test}))

    # GYU code
    # predicted = tf.arg_max(hypo,1)
    # acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predict = sess.run(hypothesis,feed_dict={x:x_test})
#     print(predict,sess.run(tf.argmax(predict,axis=1)))
'''