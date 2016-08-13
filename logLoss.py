import tensorflow as tf

def logLoss(preds, ys):
    with tf.name_scope("logLoss") as scope:
        epsilon = 1e-7
        preds2 = tf.maximum(preds, epsilon)
        preds2 = tf.minimum(preds2, 1-epsilon)
        posLogLoss = tf.mul(ys, tf.log(preds2))
        negLogLoss = tf.mul(1 - ys, tf.log(1 - preds2))
        return -tf.reduce_mean(tf.add(posLogLoss, negLogLoss))
