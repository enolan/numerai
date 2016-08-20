import ast
import random
import string
import os
import tensorflow as tf
import shutil
from loadData import *
from logLoss import *
from timer import Timer
import sys


def train(predictor, modelName, hyperparams):
    timer = Timer()

    with tf.Session() as sess:
        tf.set_random_seed(19900515)
        minibatch_size = int(hyperparams["minibatch_size"])

        ys = tf.placeholder(tf.float32, shape=[None, 1])
        xs = tf.placeholder(tf.float32, shape=[None, 21])
        isTraining = tf.placeholder(tf.float32, shape=[])

        preds, preDescentOp = predictor(xs, isTraining, hyperparams)
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
        paramPath = modelName

        tf.initialize_all_variables().run(session=sess)

        print("loading params")
        maybeCheckpoint = tf.train.latest_checkpoint("params", latest_filename=modelName + "-latest")
        if maybeCheckpoint != None:
            saver.restore(sess, maybeCheckpoint)
        else:
            print("no checkpoint found")

        timer.measure("initialization")

        min_test_loss = None
        best_iter = 0

        for i in range(int(trainData.shape[0]/minibatch_size*200)):
            batchFeatures, batchYs = getMinibatch(minibatch_size)
            preDescentOp.run(session=sess)
            opt_op.run(session=sess, feed_dict={xs: batchFeatures, ys: batchYs, isTraining: 1})
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
                    return min_test_loss, testLoss, trainLoss, True

                saver.save(sess, paramPath, global_step=step_count, latest_filename=modelName + "-latest")
                if best_iter == step_count:
                    file_suffixes = ["", ".meta"]
                    for suff in file_suffixes:
                        shutil.copyfile(paramPath + "-" + str(step_count) + suff,
                                        paramPath + "-" + suff + "best")

                timer.measure("saving")
                print(
                    'Batch {:6}, epoch {:f}, train loss {:f}, test loss {:f}'
                    .format(step_count, step_count*minibatch_size/trainData.shape[0], trainLoss, testLoss))
    return min_test_loss, testLoss, trainLoss, False

def writePredictions(predictor, modelName):
    sess = tf.InteractiveSession()

    hyperparams = {}
    with open(sys.argv[2]) as h:
        str_params = h.read()
        unresolved_params = ast.literal_eval(str_params)
        hyperparams = resolve_hyper_fns(unresolved_params)
    xs = tf.placeholder(tf.float32, shape=[None, 21])
    preds, _preDescentOp = predictor(xs, 0, hyperparams)
    tf.initialize_all_variables().run()

    tf.train.Saver().restore(sess, sys.argv[3])

    predsArr = preds.eval(feed_dict = {xs: getTournamentData()})
    out = numpy.concatenate((getTournamentTids(), predsArr), 1)
    numpy.savetxt(modelName + "-out.csv", out, delimiter=',', fmt=["%i", "%f"], comments="", header="\"t_id\",\"probability\"")

def resolve_hyper_fns(hyperparams):
    res = {}
    for k, v in hyperparams.items():
        if type(v) == dict:
            res[k] = resolve_hyper_fns(v)
        elif v == "relu":
            res[k] = tf.nn.relu
        elif v == "sigmoid":
            res[k] = tf.sigmoid
        else:
            res[k] = v
    return res

def sample_hyperparams(hyperparam_search_dict):
    res = {}
    for k, v in hyperparam_search_dict.items():
        ty = type(v)
        if ty == dict:
            res[k] = sample_hyperparams(v)
        elif ty == tuple:
            if len(v) == 2:
                if type(v[0]) == float and type(v[1]) == float:
                    res[k] = random.uniform(v[0],v[1])
                elif type(v[0]) == int and type(v[1]) == int:
                    res[k] = random.randrange(v[0],v[1])
                else:
                    print("bad type in " + k)
                    exit(1)
            else:
                print("bad range in " + k)
                exit(1)
        elif ty == list:
            res[k] = random.choice(v)
        else:
            res[k] = v
    return res

def gen_rand_id():
    id = ""
    for _ in range (32):
        id += random.choice(string.ascii_letters + string.digits)
    return id

def mk_cols(in_dict):
    cols = []
    for k in sorted(in_dict):
        if type(in_dict[k]) == dict:
            sub_dict_cols = mk_cols(in_dict[k])
            for label in sub_dict_cols:
                cols.append(k + "-" + label)
        else:
            cols.append(k)
    return cols

def csv_dict_vals(in_dict):
    res = ""
    for k in sorted(in_dict):
        if type(in_dict[k]) == dict:
            res += csv_dict_vals(in_dict[k])
        else:
            res += str(in_dict[k]) + ","
    return res

def hypersearch(predictor, modelName, hyperparam_search_dict):
    with open(modelName + "-search.csv", 'w', buffering=1) as csv:
        csv.write("id,min test loss,final test loss,final train loss,diff,finished early,")
        for label in mk_cols(hyperparam_search_dict):
            csv.write(label + ",")
        csv.write("\n")
        while True:
            run_id = gen_rand_id()
            run_path = modelName + "-search/" + run_id
            os.makedirs(run_path)
            os.chdir(run_path)
            sampled_params = sample_hyperparams(hyperparam_search_dict)
            sampled_params_resolved = resolve_hyper_fns(sampled_params)
            with open("hyperparams", 'w') as h:
                h.write(str(sampled_params))
            print("starting run with params {}".format(sampled_params))
            min_test_loss, final_test_loss, final_train_loss, finished_early = train(predictor, modelName, sampled_params_resolved)
            tf.reset_default_graph()
            csv.write("{},{},{},{},{},{},".
                      format(run_id, min_test_loss, final_test_loss,
                             final_train_loss, final_test_loss - final_train_loss, finished_early))
            csv.write(csv_dict_vals(sampled_params_resolved) + "\n")
            os.chdir("../..")

def go(predictor, modelName, hyperparam_search_dict):
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("bad args")
        exit(1)
    elif sys.argv[1] == "train":
        handle = open(sys.argv[2], 'r')
        hyperparams_str = handle.read()
        handle.close()
        hyperparams_unresolved = ast.literal_eval(hyperparams_str)
        hyperparams = resolve_hyper_fns(hyperparams_unresolved)
        train(predictor, modelName, hyperparams)
    elif sys.argv[1] == "hypersearch":
        hypersearch(predictor, modelName, hyperparam_search_dict)
    elif sys.argv[1] == "predict":
        writePredictions(predictor, modelName)
