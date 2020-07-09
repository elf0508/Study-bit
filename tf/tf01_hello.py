import tensorflow as tf

print(tf.__version__)  # 1.14.0

hello = tf.constant("Hello World")  # constant : 바뀌지 않는 상수

# print(hello)  # Tensor("Const:0", shape=(), dtype=string) <-- 텐서의 자료형

# 사람 눈에 보이게 하려면 Session을 통과시켜야 한다.

sess = tf.Session()
print(sess.run(hello))   # 'Hello World'

'''
  
import tensorflow as tf
print(tf.__version__)               # 1.14.0

hello = tf.constant("Hello World")  # constant : 상수=바뀌지 않는 값

print(hello)                        # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()                 # 상수를 사람 눈에 보여주기 위해서는 Session을 통과해야 한다.
                                    # 기계에서 연산되어 기계가 잘 알아듣도록 한다.
print(sess.run(hello))              # 'Hello World'

'''




