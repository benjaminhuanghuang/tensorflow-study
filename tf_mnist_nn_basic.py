'''

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# One hot encoded format means that our data consists of a vector like this with nine entries.
# data will be labeled with a 1 corresponding to the column for that label and then 0 otherwise.
# For 0, the label is [1,0,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# put 100 image each batch
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

# Place holders as the input
# mnist data image of shape 28 * 28 = 784
x = tf.placeholder(tf.float32, [None, 784])
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10])

# create neural network without hidden layer
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.matmul(x, W) + b

# Loss function
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
loss = tf.reduce_mean(tf.square(y - prediction))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
train = optimizer.minimize(loss)

# Accuracy, [True, False, True...]
# tf.argmax(y, 1) return index of max value in row
corrent_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):   # train 21 times
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter", epoch, "testing accuract", acc)