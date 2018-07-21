import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 200 number beween -0.5 to 0.5 in 200 rows, 1 col
# [:, np.newaxis] use all elements from the first dimension and add a second dimension:
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
#print(x_data)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 1 hidden layer create neural network
weights_l1 = tf.Variable(tf.random_normal([1, 10]))   # 1 row, 10 col
biases_l1 = tf.Variable(tf.zeros([1, 10]))  # 1 row ,10 col, init value is 0

l1 = tf.nn.tanh(tf.matmul(x, weights_l1) + biases_l1)

# output layer
weights_output = tf.Variable(tf.random_normal([10, 1]))   # 10 row, 1 col
biases_output = tf.Variable(tf.zeros([1, 1]))  # 1 row ,1 col, init value is 0

prediciton = tf.nn.tanh(tf.matmul(l1, weights_output) + biases_output)

#
loss = tf.reduce_mean(tf.square(prediciton - y))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    prediciton_value = sess.run(prediciton, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediciton_value, 'r-', lw=3)
    plt.show()
