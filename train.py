import tensorflow as tf
import shutil
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
    isTraining = tf.placeholder(tf.float32, shape=[])

    preds, preDescentOp = predictor(xs, isTraining)
    loss = logLoss(preds, ys)

    global_step = tf.Variable(0)
    opt = tf.train.AdamOptimizer()
    opt_op = opt.minimize(loss, global_step = global_step)

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

    min_test_loss = None
    best_iter = 0

    for i in range(int(trainData.shape[0]/minibatchSize*2000000)):
        batchFeatures, batchYs = getMinibatch()
        preDescentOp.run()
        opt_op.run(feed_dict={xs: batchFeatures, ys: batchYs, isTraining: 1})
        if i % 2000 == 0:
            timer.measure("2000 descent iterations")

            trainLoss, trainLossSummary, step_count = sess.run(
                [loss, merged, global_step],
                feed_dict = {ys: batchYs, xs: batchFeatures, isTraining: 0})
            trainWriter.add_summary(trainLossSummary, step_count)
            timer.measure("train loss computation")

            testFeatures, testYs = getTestFeatures(), getTestYs()
            testLoss, testLossSummary = sess.run(
                [loss, merged],
                feed_dict = {ys: testYs, xs: testFeatures, isTraining: 0})
            testWriter.add_summary(testLossSummary, step_count)
            timer.measure("test loss computation")

            if min_test_loss == None or testLoss < min_test_loss:
                min_test_loss = testLoss
                best_iter = step_count
                print("updating min_test_loss = {:f}, best_iter = {}".
                      format(min_test_loss, best_iter))

            elif best_iter + 10000 <= step_count:
                print("No improvement in 10000 iterations, min_test_loss = {:f}".
                      format(min_test_loss))
                exit(0)

            saver.save(sess, paramPath, global_step=step_count, latest_filename=modelName + "-latest")
            if best_iter == step_count:
                file_suffixes = ["", ".meta"]
                for suff in file_suffixes:
                    shutil.copyfile(paramPath + "-" + str(step_count) + suff,
                                    paramPath + "-" + suff + "best")

            timer.measure("saving")
            print(
                'Batch {:6}, epoch {:f}, train loss {:f}, test loss {:f}'
                .format(step_count, step_count*minibatchSize/trainData.shape[0], trainLoss, testLoss))

def writePredictions(predictor, modelName):
    sess = tf.InteractiveSession()

    xs = tf.placeholder(tf.float32, shape=[None, 21])
    preds, _preDescentOp = predictor(xs, 0)
    tf.initialize_all_variables().run()

    maybeCheckpoint = tf.train.latest_checkpoint("params", latest_filename=modelName + "-latest")
    if maybeCheckpoint != None:
        tf.train.Saver().restore(sess, maybeCheckpoint)
    else:
        print("no checkpoint found")
        exit(1)

    predsArr = preds.eval(feed_dict = {xs: getTournamentData()})
    out = numpy.concatenate((getTournamentTids(), predsArr), 1)
    numpy.savetxt(modelName + "-out.csv", out, delimiter=',', fmt=["%i", "%f"], comments="", header="\"t_id\",\"probability\"")

def go(predictor, modelName):
    if len(sys.argv) != 2:
        print("bad args")
        exit(1)
    elif sys.argv[1] == "train":
        train(predictor, modelName)
    elif sys.argv[1] == "predict":
        writePredictions(predictor, modelName)
