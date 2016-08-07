import tensorflow as tf
from loadData import *
from logLoss import *


def train(predictor):
    sess = tf.InteractiveSession()
    tf.set_random_seed(19900515)

    ys = tf.placeholder(tf.float32, shape=[None, 1])
    xs = tf.placeholder(tf.float32, shape=[None, 21])

    preds = predictor(xs=xs)

    opt = tf.train.GradientDescentOptimizer(0.1)
    opt_op = opt.minimize(logLoss(preds, ys))

    tf.initialize_all_variables().run()

    for i in range(1000000):
        batchFeatures, batchYs = getMinibatch()
        opt_op.run(feed_dict={xs: batchFeatures, ys: batchYs})
        if i % 100 == 0:
            trainLoss = logLoss(preds, ys).eval(
                feed_dict={ys: batchYs, xs: batchFeatures})
            testFeatures, testYs = getTestFeatures(), getTestYs()
            testLoss = logLoss(preds, ys).eval(
                feed_dict={ys: testYs, xs: testFeatures})
            print(
                'Batch {:6}, epoch {:f}, train loss {:f}, test loss {:f}'
                .format(i, i/trainData.shape[0], trainLoss, testLoss))
