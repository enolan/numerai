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
    isTraining = tf.placeholder(tf.float32, shape=[])

    preds, preDescentOp = predictor(xs, isTraining)
    loss = logLoss(preds, ys)

    global_step = tf.Variable(0)
    # learning_rate = tf.train.exponential_decay(0.001, global_step, trainData.shape[0]/minibatchSize*20, 0.95, staircase=True)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.95) # tf.train.RMSPropOptimizer(learning_rate, momentum=0.99)
    # opt = tf.train.RMSPropOptimizer(learning_rate)
    opt = tf.train.AdamOptimizer()
    opt_op = opt.minimize(loss, global_step = global_step)

    # tf.scalar_summary("learning rate", learning_rate)
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

    for i in range(int(trainData.shape[0]/minibatchSize*2000000)):
        batchFeatures, batchYs = getMinibatch()
        # grads = opt.compute_gradients(loss)
        # for g in grads:
        #     print(g[0].eval(feed_dict={xs: batchFeatures, ys:batchYs, isTraining: 1}))
        #     print(g[1].eval())
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

            saver.save(sess, paramPath, global_step=i, latest_filename=modelName + "-latest")
            timer.measure("saving")
            print(
                'Batch {:6}, epoch {:f}, train loss {:f}, test loss {:f}'
                .format(step_count, step_count*minibatchSize/trainData.shape[0], trainLoss, testLoss))

def writePredictions(predictor, modelName):
    sess = tf.InteractiveSession()

    xs = tf.placeholder(tf.float32, shape=[None, 21])
    preds = predictor(xs)
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
