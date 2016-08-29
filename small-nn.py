import copy
import tensorflow as tf

import mklayer as l
import train

def predict(xs, isTraining, hyperparams):
    hyp = copy.deepcopy(hyperparams)
    hyp["hidden1"] = hyperparams["hidden"]
    inputDropped = tf.nn.dropout(xs, l.compute_keep(isTraining, hyp["input_keep_prob"]))
    hidden1, hidden1Op = l.mklayer(inputDropped, isTraining, "hidden1", hyp)
    out, outOp = l.mklayer(hidden1, "output", hyp)
    combinedOp = tf.group(hidden1Op, outOp)
    return out, combinedOp

hidden_params = {
                 "keep_prob": 1.,
                 "bias_init": 0.0,
                 "max_norm":  100000,
                 "outputs":   (10,100),
                 "act_fn":    "relu"}

train.go(predict, "smallnn",
         {
             "hidden": hidden_params,
             "output": {
                 "keep_prob": 1.0,
                 "bias_init": 0.0,
                 "max_norm":  1000000000,
                 "outputs":   1,
                 "act_fn":    "sigmoid"},
             "input_keep_prob": 1.,
             "minibatch_size" : (1,200)
          })
