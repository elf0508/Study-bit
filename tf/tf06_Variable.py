import tensorflow as tf

tf.set_random_seed(777)


W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

print(W)  # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

W = tf.Variable([0.3], tf.float32)
# W = tf.Variable([10], tf.float32)
# W = tf.Variable([777], tf.float32)

print("====== sess.run(W) ==========")

sess = tf.Session()

sess.run(tf.global_variables_initializer()) # 변수 선언(변수 초기화)

aaa = sess.run(W)

print("aaa : ", aaa)  # aaa :  [0.3]

sess.close()   # sess.run 사용시, 마지막에 sess.close() 필수!!!!

#####################################
print("====== tf.InteractiveSession() ==========")

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

bbb = W.eval()

print("bbb : ", bbb)  # bbb :  [0.3]

sess.close()

##########################################
print("====== W.eval(session = sess) ==========")

sess = tf.Session()

sess.run(tf.global_variables_initializer())

ccc = W.eval(session = sess)

print("ccc : ", ccc)  # ccc :  [0.3]

sess.close()












