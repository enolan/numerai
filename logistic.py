import tensorflow as tf
from train import *

weights = tf.Variable(tf.random_uniform([21, 1], dtype=tf.float32))

def predict(xs):
    return 1 / (1 + tf.exp(-tf.matmul(xs, weights)))

train(predict)
