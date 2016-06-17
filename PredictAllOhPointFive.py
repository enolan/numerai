import tensorflow as tf
import numpy

from loadData import *
from logLoss import *

sess = tf.Session()

ys = tf.placeholder(tf.float32)
preds = tf.fill(tf.shape(ys), 0.5)
logLoss = logLoss(preds, ys)

print(logLoss.eval(feed_dict={ys: getTrainYs()}, session=sess))
