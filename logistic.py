import tensorflow as tf
from train import *

weights = tf.Variable(tf.random_uniform([21, 1], minval=-0.1, maxval=0.1, dtype=tf.float32))
tf.histogram_summary("weights", weights)

def predict(xs, _isTraining, _hyperparams):
    return 1 / (1 + tf.exp(-tf.matmul(xs, weights))), tf.no_op()

go(predict, "logistic", False, {'minibatch_size': 20})
