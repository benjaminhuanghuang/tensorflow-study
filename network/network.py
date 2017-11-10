import tensorflow as tf
from tensorflow.contrib import learn

class Network():
  def __init__(self, num_hidden, batch_size):
    '''
    @num_hidden: node count of hidden layer
    @ batch_size: load data in batch
    '''
    self.batch_size = batch_size
  
    # Hyper parameters
    self.num_hidden = num_hidden
    
    # Graph related
    self.graph = tf.Graph()
    self.tf_train_smaples = None
    self.tf_train_labels = None
    self.tf_test_samples = None
    self.tf_test_labels = None
    self.test_prediction = None


  def define_graph(self):
    with self.graph.as_default():
      self.tf_train_samples = tf.placeholder(tf.float32, shape=(self.batch_size, image_size, image_size, num_channels))
			self.tf_train_labels  = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))
			self.tf_test_samples  = tf.placeholder(tf.float32, shape=(self.test_batch_size, image_size, image_size, num_channels))

			# fully connected layer 1, fully connected
			fc1_weights = tf.Variable(
				tf.truncated_normal([image_size * image_size, self.num_hidden], stddev=0.1)
			)
			fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]))

			# fully connected layer 2 --> output layer
			fc2_weights = tf.Variable(
				tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1)
			)
			fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

			# 想在来定义图谱的运算
			def model(data):
				# fully connected layer 1
				shape = data.get_shape().as_list()
				print(data.get_shape(), shape)
				reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
				print(reshape.get_shape(), fc1_weights.get_shape(), fc1_biases.get_shape())
				hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

				# fully connected layer 2
				return tf.matmul(hidden, fc2_weights) + fc2_biases

			# Training computation.
			logits = model(self.tf_train_samples)
			self.loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels)
			)

			# Optimizer.
			self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

			# Predictions for the training, validation, and test data.
			self.train_prediction = tf.nn.softmax(logits)
			
  def train(self):
    pass

  def test(self):
    pass
  
  def accuracy(self):
    pass