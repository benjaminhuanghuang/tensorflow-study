'''
    Reference: http://blog.csdn.net/u010858605/article/details/69830657
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# One hot encoded format means that our data consists of a vector like this with nine entries.
# data will be labeled with a 1 corresponding to the column for that label and then 0 otherwise.
# For 0, the label is [1,0,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('Training set:')   # 55000
print(mnist.train.images.shape,mnist.train.labels.shape)
print('Testing set: ')   # 10000
print(mnist.test.images.shape,mnist.test.labels.shape)
print('Validation set: ')  # 5000
print(mnist.validation.images.shape,mnist.validation.labels.shape)

# the size of picture is 28 * 28 = 784
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))   
b = tf.Variable(tf.zeros([10]))   
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Training
y_ = tf.placeholder(tf.float32, [None,10])
# Cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# run
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Evaluate
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy: ')
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))