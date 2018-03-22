'''

'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# One hot encoded format means that our data consists of a vector like this with nine entries.
# data will be labeled with a 1 corresponding to the column for that label and then 0 otherwise.
# For 0, the label is [1,0,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('type of mnist', type(mnist))

print('mnist.train.images', mnist.train.images)

print('mnist.train.num_examples', mnist.train.num_examples)
print('mnist.test.num_examples', mnist.test.num_examples)

single_image = mnist.train.images[1].reshape(28, 28)
print('max, min value in image', single_image.max(), single_image.min())
plt.imshow(single_image, cmap='gist_gray')
plt.show()

# Place holders as the input
# mnist data image of shape 28 * 28 = 784
x = tf.placeholder(tf.float32, [None, 784])
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10])

y_true = tf.placeholder(tf.float32, [None, 10])

# Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Graph operations
y = tf.matmul(x, W) + b

# Loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # Evaluate the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    # [True, False, True...]
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(acc, feed_dict={
          x: mnist.test.images, y_true: mnist.test.labels}))


# Set model weight
