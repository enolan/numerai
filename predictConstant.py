
# coding: utf-8

# In[1]:

import tensorflow as tf
from loadData import *
from logLoss import *
sess = tf.InteractiveSession()


# In[2]:

ysDat = getTrainYs()
ys = tf.placeholder(tf.float32)
always = tf.Variable(0.1, dtype=tf.float32)
preds = tf.fill(tf.shape(ys), 1 * always)
ysDat.size


# In[3]:

opt = tf.train.GradientDescentOptimizer(0.1)
opt_op = opt.minimize(logLoss(preds, ys))


# In[4]:

tf.initialize_all_variables().run()


# In[5]:

for x in range(0,100):
    opt_op.run(feed_dict={ys:ysDat})
    print(logLoss(preds, ys).eval(feed_dict={ys:ysDat}))


# In[6]:

always.eval()


# In[ ]:



