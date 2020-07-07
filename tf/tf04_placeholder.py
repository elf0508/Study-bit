# placeholder : input 과 비슷
# + feed_dict 

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)  # constant : 변하지 않는 상수
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)

b = tf.placeholder(tf.float32)

adder_node = a + b  # 히든 레이어

# sess.run : 결과값을 보는것  feed_dict : 값이 들어간다
#               a               b
print(sess.run(adder_node, feed_dict = {a:3, b:4.5}))   # 7.5  인풋 레이어

print(sess.run(adder_node, feed_dict = {a:[1, 3], b:[2, 4]}))   # [3. 7.]

add_and_triple = adder_node * 3   # 히든 레이어

print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))   # 22.5   아웃풋

# feed_dict : 값이 들어간다

# feed 값은 일시적으로 연산의 출력값을 입력한 tensor 값으로 대체한다. 
# feed 데이터는 run()으로 전달되어서 run()의 변수로만 사용된다. 
# 가장 일반적인 사용방법은 tf.placeholder()를 사용해서 
# 노드 (히든 레이어)에서 "feed" 작업으로 지정해 주는 것입니다.

'''
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)    # 변하지 않는 값
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) 

sess = tf.Session()

a = tf.placeholder(tf.float32)          # input와 비슷한 개념 
b = tf.placeholder(tf.float32)          # : sess.run할 때 fedd_dict로 값을 집어넣어준다.

adder_node = a + b
                                                        # feead bit 집어넣을 값들
print(sess.run(adder_node, feed_dict = {a:3, b:4.5}))   # dict안에 들어간 수를 빼와서 사용
print(sess.run(adder_node, feed_dict = {a:[1, 3], b:[2, 4]})) # [3. 7.]

add_and_triple = adder_node * 3                         # (a + b) *3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5})) # 22.5

'''
