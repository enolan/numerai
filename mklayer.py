import tensorflow as tf

def compute_keep(isTraining, prob):
    with tf.name_scope("computeKeep"):
        return (1 - isTraining * (1 - prob))

def mklayer(xs, name, hyperparams):
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
