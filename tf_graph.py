import numpy as np
import tensorflow as tf

np.random.seed(101)
tf.set_random_seed(101)

# 5 * 5 unifrom random 
rand_a = np.random.uniform(0, 100, (5,5))
rand_b = np.random.uniform(0,100, (5,1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a + b
mul_op = a + b
with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict ={a:rand_a, b:rand_b})
    