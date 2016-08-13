import tensorflow as tf
import train

def computeKeep(isTraining, prob):
    return (1 - isTraining * (1 - prob))

def mkLayer(xs, outputs, keep_prob, act_fn, maxNormConstraint, name):
    numInputs = xs.get_shape().as_list()[-1]
    weight_shape = [numInputs, outputs]
    weights = tf.Variable(tf.truncated_normal(weight_shape, stddev=tf.sqrt(2. / tf.to_float(numInputs)), dtype=tf.float32, name=name + "-weights"))
    tf.histogram_summary(name + "-weights", weights)
    norms = tf.sqrt(tf.reduce_sum(weights ** 2, reduction_indices=1))
    tf.histogram_summary(name + "-norm-weighs", norms)
    maxNorm = tf.reduce_max(norms)
    tf.scalar_summary(name + "-max-norm", tf.reduce_max(norms))
    biases = tf.Variable(tf.constant(0.01, shape = [outputs]))
    tf.histogram_summary(name + "biases", biases)
    sums = tf.matmul(xs, weights) + biases
    sumsDropped = tf.nn.dropout(sums, keep_prob)
    activations = act_fn(sumsDropped)
    tf.histogram_summary(name + "-activations", activations)
    maxNormOp = weights.assign(tf.clip_by_norm(weights, maxNormConstraint, axes=[1]))
    return activations, maxNormOp # tf.Print(activations, [maxNorm])
    # return tf.Print(tf.nn.relu(sumsDropped), [sumsDropped], summarize=1000000000)

def predict(xs, isTraining):
    inputDropped = tf.nn.dropout(xs, computeKeep(isTraining, 0.95))
    hidden1, hidden1Op = mkLayer(inputDropped, 128, (computeKeep(isTraining, 0.5)), tf.nn.relu, 4, "hidden1")
#    hidden = tf.contrib.layers.fully_connected(inputDropped, num_outputs = 256, activation_fn = tf.nn.relu)
    # hidden2, hidden2Op = mkLayer(hidden1, 512, computeKeep(isTraining, 1), tf.nn.relu, 9.11, "hidden2")
    out, outOp = mkLayer(hidden1, 1, 1, tf.sigmoid, 100000, "output")
    combinedOp = tf.group(hidden1Op, outOp)
    return out, combinedOp

train.go(predict, "smallnn")
