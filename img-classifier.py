'''
Tensorflow in 5 minutes
https://www.youtube.com/watch?v=2FmcHiLCwTU

1. Download dataset
2. splits it into training and testing data

'''

# input MNIST data
import input_data
mnist = input_data.read_data_sets("/data", one_hot=True)

import tensorflow as tf
# set parameters
# how fast we want to update our weight
# If learning rate is too big our model might skip the optimal solution
# If it is too small we might need too may iteration to converge on the
# best result.
learing_rate = 0.01

training_iteration = 30
batch_size = 100
display_step = 2

# mnist data image of shape 28*28 = 784
x = tf.placeholder('float', [None, 784])
# 0-9 digits recognition => 10 classes
y = tf.placeholder('float', [None, 10])

with tf.name_scope('Wx_b') as scope:
  # Construct a linear model
  model = tf.nn.softmax(tf.matmul(x, X) + b)  # softmax

# Add summary ops to collect data
w_h = tf.histogram_summary('weights', W)
b_b = tf.histogram_summary('biases', b)

# more name scopes will clean up graph representation
with tf.name_scope('cost_function') as scope:
  # minimize error using cross entropy
  # cross entropy
  const_function = -tf.reduce_sum(y * tf.log(model))
  # create a summary to monitor the cost function
  tf.scalar_summary('cost_function', cost_function)

with tf.name_scope('train') as scope:
  # gradient descent
  optimizer = tf.train.GradientDescentOptimizer(learing_rate).minimize(const_function)


# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
  sess.run(init)

  # set the logs writer to the folder /tmp/tensorflow_logs
  # create data to visualize in tensorboard:
  # $ tensorboar --logdir="logs"
  summary_writer = tf.train.SummaryWriter('logs/', graph_def = sess.graph_def)
  
  # training cycle
  for iteration in range(training_iteration):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # loop over all batches
    for i in range(total_batch):
      batch_xs, batch_ys = nmist.train.next_batch(batch_size)
      # fit training using batch data
      sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
      # compute the average loss
      avg_cost += sess.run(const_function, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch
      # write logs for each iteration
      summary_str = sess.run(merged_summary_op, feed_dict = {x: batch_xs, y: batch_ys})
      summary_writer.add_summary(summary_str, iteration * total_batch + i)

    # Display log per iteration step
    if iteration % display_step == 0:
      print "Iteration: ", '%04d' % (iteration + 1), "cost =", "{:.9f}".format(avg_cost)

print "Training completed"

# Test the model
predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean( tf.cast(predictions, "float"))
print "Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
