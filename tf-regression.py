from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))  # a tensor with all elements set to zero.

y = Weights * x_data + biases    # Task: Find Weights and biases to match y_data

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # learning rate is 0.5
train = optimizer.minimize(loss)   # use optimizer to minimize loss

init = tf.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.Session()

sess.run(init)

for step in range(401):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
