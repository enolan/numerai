import tensorflow as tf
from loadData import *
from logLoss import *

sess = tf.InteractiveSession()

ysDat = getTrainYs()
ys = tf.placeholder(tf.float32)
always = tf.Variable(0.1, dtype=tf.float32)
preds = tf.fill(tf.shape(ys), always)

opt = tf.train.GradientDescentOptimizer(0.1)
opt_op = opt.minimize(logLoss(preds, ys))

tf.initialize_all_variables().run()

for x in range(100):
    opt_op.run(feed_dict={ys:ysDat})
    print(logLoss(preds, ys).eval(feed_dict={ys:ysDat}))

print("Best constant prediction:")
print(always.eval())
