import numpy as np
import tensorflow as tf

np.random.seed(101)
tf.set_random_seed(101)

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))

W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))
xW = tf.matmul(x, W)
z = tf.add(xW, b)
# Activation function
a = tf.sigmoid(z)

init = tf.global_variables_initializer()
with tf.session as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})

print(layer_out)