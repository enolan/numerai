import tensorflow as tf
from loadData import *
from logLoss import *


def train(predictor):
    sess = tf.InteractiveSession()
    tf.set_random_seed(19900515)

    ys = tf.placeholder(tf.float32, shape=[None, 1])
    xs = tf.placeholder(tf.float32, shape=[None, 21])

    preds = predictor(xs=xs)

    opt = tf.train.RMSPropOptimizer(0.03)
    opt_op = opt.minimize(logLoss(preds, ys))

    loss = logLoss(preds, ys)

    tf.scalar_summary("loss", loss)
    merged = tf.merge_all_summaries()
    trainWriter = tf.train.SummaryWriter("logs/train", sess.graph)
    testWriter = tf.train.SummaryWriter("logs/test")

    tf.initialize_all_variables().run()

    for i in range(trainData.shape[0]*5):
        batchFeatures, batchYs = getMinibatch()
        opt_op.run(feed_dict={xs: batchFeatures, ys: batchYs})
        if i % 100 == 0:
            trainLoss, trainLossSummary = sess.run(
                [loss, merged],
                feed_dict = {ys: batchYs, xs: batchFeatures})
            trainWriter.add_summary(trainLossSummary, i)
            testFeatures, testYs = getTestFeatures(), getTestYs()
            testLoss, testLossSummary = sess.run(
                [loss, merged],
                feed_dict = {ys: testYs, xs: testFeatures})
            testWriter.add_summary(testLossSummary, i)
            print(
                'Batch {:6}, epoch {:f}, train loss {:f}, test loss {:f}'
                .format(i, i/trainData.shape[0], trainLoss, testLoss))
