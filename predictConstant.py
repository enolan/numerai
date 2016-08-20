import tensorflow as tf
from train import *

always = tf.Variable(0.1, dtype=tf.float32)
def predict(xs, _isTraining, _hyperparams):
    return tf.fill([tf.shape(xs)[0], 1], always), tf.no_op()

go(predict, "constant", False, {})
