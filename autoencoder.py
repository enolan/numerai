from timer import Timer
timer = Timer()

import tensorflow as tf
timer.measure("import tensorflow")

import itertools
import numpy

import loadData
from hypersearch import hypersearch, resolve_hyper_fns
import mklayer
timer.measure("import rest")

hyp = {"minibatch_size": 50,
       "input_keep_prob": 0.8,
       "autoenc": {
           "hidden": {"outputs": 2048,
                      "bias_init": 0.,
                      "keep_prob": 1.,
                      "max_norm": None,
                      "act_fn": "relu"},
           "output": {"outputs": 21,
                      "bias_init": 0.,
                      "keep_prob": 1.,
                      "max_norm": None,
                      "act_fn": "identity"}
       }}
hyp = resolve_hyper_fns(hyp)


def simple_dropout(xs, keep_prob, on):
    with tf.name_scope("simple_dropout"):
        random_tensor = tf.maximum(keep_prob, 1 - on)
        random_tensor += tf.random_uniform(tf.shape(xs))
        mask = tf.floor(random_tensor)
        return xs * mask


def mk_autoenc_layer(xs, hyp, num):
    name = str(num) + "-"
    inner_hyp = {}
    for k, v in hyp["autoenc"].items():
        inner_hyp[name + k] = v
    hidden, _ = mklayer.mklayer(xs, 0, name + "hidden", inner_hyp)
    output, _ = mklayer.mklayer(hidden, 0, name + "output", inner_hyp)
    return hidden, output


def output_features(layer):
    train_data = numpy.loadtxt(
        loadData.dataDir + "/numerai_training_data.csv",
        skiprows=1,
        delimiter=',')
    train_extra_features = sess.run(
        layer,
        feed_dict={xs: loadData.getFeatures(train_data) - loadData.means_train,
                   })
    tournament_extra_features = sess.run(
        layer,
        feed_dict={xs: loadData.getTournamentData() - loadData.means_train, })
    print("tournament_extra_features.shape = " + str(
        tournament_extra_features.shape))
    print("train_extra_features.shape = " + str(train_extra_features.shape))
    train_concatenated_features = numpy.concatenate(
        (loadData.getFeatures(train_data), train_extra_features,
         loadData.getYs(train_data)), 1)
    tournament_concatenated_features = numpy.concatenate(
        (loadData.tournamentDataIn_orig, tournament_extra_features), 1)
    numpy.savetxt(
        "autoenc-out-train.csv",
        train_concatenated_features,
        delimiter=',',
        comments='',
        fmt='%1.15f')
    numpy.savetxt(
        "autoenc-out-tournament.csv",
        tournament_concatenated_features,
        delimiter=',',
        comments='',
        fmt='%1.15f')


iters_per_epoch = loadData.trainData.shape[0] / hyp["minibatch_size"]
# rho = hyp["rho"]

xs = tf.placeholder(tf.float32, shape=[None, 21], name="xs")
global_step = tf.Variable(0, name="global_step")

autoencs = []
for i in range(4):
    with tf.name_scope("autoenc-layer-" + str(i)):
        autoencs.append({})
        if i == 0:
            input_t = xs
        else:
            input_t = autoencs[i - 1]["hidden"]
        this_is_training = tf.Variable(
            1., trainable=False, name="this_is_training-" + str(i))
        tf.scalar_summary(str(i) + '-this_is_training', this_is_training)
        autoencs[i]["is_training"] = this_is_training
        input_dropped = simple_dropout(input_t, 0.8, this_is_training)
        hidden, output = mk_autoenc_layer(input_dropped, hyp, i)
        autoencs[i]["hidden"] = hidden

        accuracy_loss = tf.contrib.losses.sum_of_squares(output, xs)
        autoencs[i]['accuracy_loss'] = accuracy_loss
        tf.scalar_summary("accuracy_loss-" + str(i), accuracy_loss)

        # with tf.name_scope("sparsity_loss"):
        #     average_activations = tf.reduce_mean(hidden, reduction_indices=1)
        #     sparsity_loss = tf.reduce_mean(rho * tf.log(
        #         rho / average_activations) + (1 - rho) * tf.log((1 - rho) / (
        #             1 - average_activations)))
        #     tf.scalar_summary("sparsity_loss-" + str(i), sparsity_loss)

        loss = accuracy_loss  #sparsity_loss + accuracy_loss
        tf.scalar_summary("loss-" + str(i), loss)
        autoencs[i]["loss"] = loss

        opt = tf.train.AdamOptimizer()
        training_vars = tf.get_collection(str(
            i) + "-hidden") + tf.get_collection(str(i) + "-output")
        opt_op = opt.minimize(
            loss, global_step=global_step, var_list=training_vars)
        autoencs[i]["opt_op"] = opt_op

merged = tf.merge_all_summaries()
log_dir = "logs/autoencoder"

timer.measure("graph construction")

with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter(log_dir + "/train", sess.graph)
    test_writer = tf.train.SummaryWriter(log_dir + "/test")

    saver = tf.train.Saver()

    tf.initialize_all_variables().run(session=sess)
    timer.measure("initialization")

    min_test_loss = None
    best_iter = 0

    for n in range(len(autoencs)):
        print("training autoencoder {}".format(n))
        best_test_loss = None
        best_test_loss_iter = None
        for i in itertools.count():
            batch_features, _ = loadData.getMinibatch(hyp["minibatch_size"])
            autoencs[n]["opt_op"].run(session=sess,
                                      feed_dict={xs: batch_features, })
            if i % 2000 == 0:
                timer.measure("2000 descent iterations")

                train_loss, train_summary, step_count = sess.run(
                    [autoencs[n]["loss"], merged, global_step],
                    feed_dict={xs: batch_features, })
                train_writer.add_summary(train_summary, step_count)
                timer.measure("train loss computation")

                test_features = loadData.getTestFeatures()
                test_loss, test_summary = sess.run(
                    [autoencs[n]["loss"], merged],
                    feed_dict={xs: test_features, })
                test_writer.add_summary(test_summary, step_count)
                timer.measure("test loss computation")
                if best_test_loss == None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_loss_iter = i
                    autoencs[n]["best_test_loss"] = best_test_loss
                    print("new best loss: {:f} on iter {}".format(
                        best_test_loss, best_test_loss_iter))
                if best_test_loss_iter + (15 * iters_per_epoch) < i:
                    print("stopping training autoencoder {}".format(n))
                    sess.run(autoencs[n]["is_training"].assign(0.))
                    test_acc_loss = sess.run(autoencs[n]["accuracy_loss"],
                                             feed_dict={xs: test_features, })
                    autoencs[n]["best_test_loss"] = test_acc_loss
                    print("accuracy loss (no randomness): {:f}".format(
                        test_acc_loss))
                    if n > 0 and test_acc_loss > autoencs[n - 1][
                            "best_test_loss"]:
                        print(
                            "accuracy worse than previous autoencoder, stopping")
                        sess.run(autoencs[n]["is_training"].assign(0.))
                        output_features(autoencs[n - 1]["hidden"])
                        exit(0)
                    break
                print(
                    "Batch {:6}, epoch {:f}, train loss {:f}, test loss {:f}".format(
                        step_count, step_count / iters_per_epoch, train_loss,
                        test_loss))
    output_features(autoencs[-1]["hidden"])
