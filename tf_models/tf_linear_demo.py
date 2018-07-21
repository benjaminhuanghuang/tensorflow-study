import numpy as np
import tensorflow as tf

x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# create a linear model
w = tf.Variable(0)
b = tf.Variable(0)

y_predict = w * x_data + b

# cost
loss = tf.reduce_mean(tf.square(y_predict - y_data))

# optimizer, learning rate - 0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)

train = optimizer.minimize(loss)
