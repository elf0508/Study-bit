# 이진분류 : sigmoid

from sklearn.datasets import load_breast_cancer
import tensorflow as tf

breast_cancer = load_breast_cancer()

x = breast_cancer.data
y = breast_cancer.target

print(x.shape)                     # (569, 30)
print(y.shape)                     # (569, )

x = tf.placeholder(tf.float32, shape=[None, ])
y = tf.placeholder(tf.float32, shape=[None, ])




