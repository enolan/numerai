import tensorflow as tf
from loadData import *
from logLoss import *
from timer import Timer
import sys


def train(predictor, modelName):
    timer = Timer()

    sess = tf.InteractiveSession()
    tf.set_random_seed(19900515)

    ys = tf.placeholder(tf.float32, shape=[None, 1])
    xs = tf.placeholder(tf.float32, shape=[None, 21])

    preds = predictor(xs=xs)

    opt = tf.train.AdadeltaOptimizer()
    opt_op = opt.minimize(logLoss(preds, ys))

    loss = logLoss(preds, ys)

    tf.scalar_summary("loss", loss)
    merged = tf.merge_all_summaries()
    logDir = "logs/" + modelName
    trainWriter = tf.train.SummaryWriter(logDir + "/train", sess.graph)
    testWriter = tf.train.SummaryWriter(logDir + "/test")

    saver = tf.train.Saver()
    paramPath = "params/" + modelName

    tf.initialize_all_variables().run()

    print("loading params")
    maybeCheckpoint = tf.train.latest_checkpoint("params", latest_filename=modelName + "-latest")
    if maybeCheckpoint != None:
        saver.restore(sess, maybeCheckpoint)
    else:
        print("no checkpoint found")

    timer.measure("initialization")

    for i in range(int(trainData.shape[0]/minibatchSize*200)):
        batchFeatures, batchYs = getMinibatch()
        opt_op.run(feed_dict={xs: batchFeatures, ys: batchYs})
        if i % 2000 == 0:
            timer.measure("2000 descent iterations")

            trainLoss, trainLossSummary = sess.run(
                [loss, merged],
                feed_dict = {ys: batchYs, xs: batchFeatures})
            trainWriter.add_summary(trainLossSummary, i)
            timer.measure("train loss computation")

            testFeatures, testYs = getTestFeatures(), getTestYs()
            testLoss, testLossSummary = sess.run(
                [loss, merged],
                feed_dict = {ys: testYs, xs: testFeatures})
            testWriter.add_summary(testLossSummary, i)
            timer.measure("test loss computation")

            saver.save(sess, paramPath, global_step=i, latest_filename=modelName + "-latest")
            timer.measure("saving")
            print(
                'Batch {:6}, epoch {:f}, train loss {:f}, test loss {:f}'
                .format(i, i*minibatchSize/trainData.shape[0], trainLoss, testLoss))

def writePredictions(predictor, modelName):
    sess = tf.InteractiveSession()

    xs = tf.placeholder(tf.float32, shape=[None, 21])
    tf.initialize_all_variables().run()

    maybeCheckpoint = tf.train.latest_checkpoint("params", latest_filename=modelName + "-latest")
    if maybeCheckpoint != None:
        tf.train.Saver().restore(sess, maybeCheckpoint)
    else:
        print("no checkpoint found")
        exit(1)

    preds = predictor(xs).eval(feed_dict = {xs: getTournamentData()})
    out = numpy.concatenate((getTournamentTids(), preds), 1)
    numpy.savetxt(modelName + "-out.csv", out, delimiter=',', fmt=["%i", "%f"], comments="", header="\"t_id\",\"probability\"")

def go(predictor, modelName):
    if len(sys.argv) != 2:
        print("bad args")
        exit(1)
    elif sys.argv[1] == "train":
        train(predictor, modelName)
    elif sys.argv[1] == "predict":
        writePredictions(predictor, modelName)
