import tensorflow as tf
import train

def predict(xs):
    hidden = tf.contrib.layers.fully_connected(xs, num_outputs = 128, activation_fn = tf.nn.relu)
    return tf.contrib.layers.fully_connected(hidden, num_outputs = 1, activation_fn = tf.sigmoid)

train.go(predict, "smallnn")
