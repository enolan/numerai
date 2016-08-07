import tensorflow as tf
from loadData import *
from logLoss import *


def train(predictor):
    sess = tf.InteractiveSession()
    tf.set_random_seed(19900515)

    ysDat = getTrainYs()
    xsDat = getTrainFeatures()

    ys = tf.placeholder(tf.float32, shape=[None, 1])
    xs = tf.placeholder(tf.float32, shape=[None, 21])

    preds = predictor(xs=xs)

    opt = tf.train.GradientDescentOptimizer(0.1)
    opt_op = opt.minimize(logLoss(preds, ys))

    tf.initialize_all_variables().run()

    for x in range(1000):
        opt_op.run(feed_dict={xs: xsDat, ys: ysDat})
        print(logLoss(preds, ys).eval(feed_dict={ys: ysDat, xs: xsDat}))
