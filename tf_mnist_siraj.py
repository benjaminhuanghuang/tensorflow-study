'''

'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# One hot encoded format means that our data consists of a vector like this with nine entries.
# data will be labeled with a 1 corresponding to the column for that label and then 0 otherwise.
# For 0, the label is [1,0,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Set parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

# TF graph input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28 * 28 = 784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Create a model

# Set model weight
W = tf.Variable(tf.zeros([784,10]))   
b = tf.Variable(tf.zeros([10]))   

with tf.name_scope('Wx_b') as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)

# Add summary ops to collect data    
w_h = tf.histogram_summary('weight', W)
b_h = tf.histogram_summary('biases', b)

# More name scopes 
with tf.name_scope('cost_function') as scope:
    # Minimize error using cross entropy
    cost_funciton = -tf.reduce_sum( y * tf.log(model))
    # Create a summary to monitor the cost function
    tf.scalar_summary('cost_function', cost_funciton)

with tf.name_scope('train') as scope:
    # Gradient descent
    optimiazer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_funciton)

init = tf.initialize_all_variables()
# Merge all summaries into a single operator
merge_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)
    # Set the logs writer to the folder 
    summary_writer = tf.train.SummaryWriter('./log', graph_def= sess.graph_def)

    # Training
    for iteration in range(training_iteration):
        avg_cost = 0
        tatal_tabch = int(mnist.train._num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimiazer, feed_dict={x: batch_xs, y: batch_y})
            avg_cost += sess.run(cost_funciton, feed_dict={x: batch_xs, y: batch_y}) / total_batch
            # Write log for each iteration
            summary_str = sess.run(merge_summary_op, feed_dict={x: batch_xs, y: batch_y})
            summary_writer.add_summary(summary_str, iteration * total_batch + 1)

        # Display logs per iteration step
        if iteration % display_step == 0:
            print("Iteration:", '%04d'%(iteration + 1), 'cost= ', "{:.9f}".format(avg_cost))

    print("Tuning completed")

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(predictions, float))
    print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))        