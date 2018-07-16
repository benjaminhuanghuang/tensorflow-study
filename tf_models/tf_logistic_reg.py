'''
https://www.bilibili.com/video/av25528464/?p=7
'''
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

active = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(active), reduction_indices=1))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Prediction
pred = tf.equal(tf.argmax(active, 1), tf.argmax(y, 1))   # index of the max elemnt in rows

# Accuracy
accr = tf.reduce_mean(tf.cast(pred, 'float'))

# Init
init = tf.global_variables_initializer()

training_epochs = 50
batch_size = 100
display_step = 5

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

    # Display
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}

        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
            % (epoch, training_epochs, avg_cost, train_acc, test_acc))