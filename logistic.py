import tensorflow as tf
from train import *
import sys

weights = tf.Variable(tf.random_uniform([21, 1], dtype=tf.float32))
tf.histogram_summary("weights", weights)

def predict(xs):
    return 1 / (1 + tf.exp(-tf.matmul(xs, weights)))

if len(sys.argv) != 2:
    print("bad args")
    exit(1)
elif sys.argv[1] == "train":
    train(predict, "logistic")
elif sys.argv[1] == "predict":
    writePredictions(predict, "logistic")
