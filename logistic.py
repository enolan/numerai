import tensorflow as tf
from loadData import *
from logLoss import *

sess = tf.InteractiveSession()

ysDat = getTrainYs()
xsDat = getTrainFeatures()

inYs = tf.placeholder(tf.float32)
ys = tf.reshape(inYs, [ysDat.shape[0], 1])

xs = tf.placeholder(tf.float32)
weights = tf.Variable(tf.random_uniform([20, 1], dtype=tf.float32))
preds = 1 / (1 + tf.exp(-tf.matmul(xs, weights)))

opt = tf.train.GradientDescentOptimizer(0.1)
opt_op = opt.minimize(logLoss(preds, ys))

tf.initialize_all_variables().run()

for x in range(1000):
    opt_op.run(feed_dict={inYs: ysDat, xs: xsDat})
    print(logLoss(preds, ys).eval(feed_dict={inYs: ysDat, xs: xsDat}))
