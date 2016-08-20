import tensorflow as tf
import copy
import train

def computeKeep(isTraining, prob):
    with tf.name_scope("computeKeep"):
        return (1 - isTraining * (1 - prob))

def mkLayer(xs, name, hyperparams):
    hyp = hyperparams[name]
    with tf.name_scope(name) as scope:
        numInputs = xs.get_shape().as_list()[-1]
        weight_shape = [numInputs, hyp["outputs"]]
        weights = tf.Variable(tf.truncated_normal(weight_shape, stddev=tf.sqrt(2. / tf.to_float(numInputs)), dtype=tf.float32, name=name + "-weights"))
        tf.histogram_summary(name + "-weights", weights)
        norms = tf.sqrt(tf.reduce_sum(weights ** 2, reduction_indices=1))
        tf.histogram_summary(name + "-norm-weights", norms)
        maxNorm = tf.reduce_max(norms)
        tf.scalar_summary(name + "-max-norm", tf.reduce_max(norms))
        biases = tf.Variable(tf.constant(hyp["bias_init"], shape = [hyp["outputs"]]))
        tf.histogram_summary(name + "-biases", biases)
        sums = tf.matmul(xs, weights) + biases
        sumsDropped = tf.nn.dropout(sums, hyp["keep_prob"])
        activations = hyp["act_fn"](sumsDropped)
        tf.histogram_summary(name + "-activations", activations)
        maxNormOp = weights.assign(tf.clip_by_norm(weights, hyp["max_norm"], axes=[1]))
        return activations, maxNormOp

def predict(xs, isTraining, hyperparams):
    hyp = copy.deepcopy(hyperparams)
    for k in ["hidden1", "hidden2", "hidden3"]:
        hyp[k] = hyperparams["hidden"]
    inputDropped = tf.nn.dropout(xs, computeKeep(isTraining, hyp["input_keep_prob"]))
    hidden1, hidden1Op = mkLayer(inputDropped, "hidden1", hyp)
#    hidden2, hidden2Op = mkLayer(hidden1, "hidden2", hyp)
#    hidden3, hidden3Op = mkLayer(hidden2, "hidden3", hyp)
    out, outOp = mkLayer(hidden1, "output", hyp)
    combinedOp = tf.group(hidden1Op)#, hidden2Op, hidden3Op, outOp)
    return out, combinedOp

hidden_params = {
                 "keep_prob": 1., #(0.2, 1.0),
                 "bias_init": 0.0,
                 "max_norm":  100000,#(2.0 ,10.0),
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
             "input_keep_prob": 1.,#(0.7, 1.0),
             "minibatch_size" : (1,200)
          })
