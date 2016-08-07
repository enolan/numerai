import tensorflow as tf
from loadData import *
from logLoss import *

sess = tf.InteractiveSession()

ysDat = getTrainYs()
xsDat = getTrainFeatures()

ys = tf.placeholder(tf.float32, shape=[None, 1])

xs = tf.placeholder(tf.float32, shape=[None, 21])
weights = tf.Variable(tf.random_uniform([21, 1], dtype=tf.float32))
preds = 1 / (1 + tf.exp(-tf.matmul(xs, weights)))

opt = tf.train.GradientDescentOptimizer(0.1)
opt_op = opt.minimize(logLoss(preds, ys))

tf.initialize_all_variables().run()

for x in range(1000):
    opt_op.run(feed_dict={ys: ysDat, xs: xsDat})
    print(logLoss(preds, ys).eval(feed_dict={ys: ysDat, xs: xsDat}))
