import tensorflow as tf

def compute_keep(isTraining, prob):
    with tf.name_scope("computeKeep"):
        return (1 - isTraining * (1 - prob))

def mklayer(xs, is_training, name, hyperparams):
    hyp = hyperparams[name]
    with tf.name_scope(name) as scope:
        numInputs = xs.get_shape().as_list()[-1]
        weight_shape = [numInputs, hyp["outputs"]]
        weights = tf.Variable(tf.truncated_normal(weight_shape, stddev=tf.sqrt(2. / tf.to_float(numInputs)), dtype=tf.float32, name=name + "-weights"))
        tf.add_to_collection(name, weights)
        tf.histogram_summary(name + "-weights", weights)
        norms = tf.sqrt(tf.reduce_sum(weights ** 2, reduction_indices=1))
        tf.histogram_summary(name + "-norm-weights", norms)
        maxNorm = tf.reduce_max(norms)
        tf.scalar_summary(name + "-max-norm", tf.reduce_max(norms))
        biases = tf.Variable(tf.constant(hyp["bias_init"], shape = [hyp["outputs"]]))
        tf.add_to_collection(name, biases)
        tf.histogram_summary(name + "-biases", biases)
        sums = tf.matmul(xs, weights) + biases
        activations = hyp["act_fn"](sums)
        activations_dropped = tf.nn.dropout(activations, compute_keep(is_training, hyp["keep_prob"]))
        tf.histogram_summary(name + "-activations", activations)
        if hyp["max_norm"] != None:
            maxNormOp = weights.assign(tf.clip_by_norm(weights, hyp["max_norm"], axes=[1]))
        else:
            maxNormOp = tf.no_op()
        return activations_dropped, maxNormOp
