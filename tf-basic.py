'''
TF Girls 修炼指南 2
https://www.youtube.com/watch?v=TrWqRMJZU8A
'''
# encoding: utf-8
from __future__ import print_function, division
import tensorflow as tf

print('Loaded TF version', tf.__version__, '\n\n')

'''
4 Types:
  Variable
  Tensor
  Graph
  Session

'''

# Symbolic programming
def Addition():
  v1 = tf.Variable(10)    # v1 is Variable
  v2 = tf.Variable(5)
  addv = v1 + v2    # Get a Tensor
	# session is a runtime
  sess = tf.Session()

	# Variable -> initialize -> Tensor
  tf.global_variables_initializer().run(session=sess)

  print('v1 + v2 =', addv.eval(session=sess))
  # or
  print('v1 + v2 =', sess.run(addv))

# Symbolic programming
def Constant():
  c1 = tf.constant(10)  # c1 is a tensor
  c2 = tf.constant(5)
  addc = c1 + c2
	
  sess = tf.Session()

  tf.global_variables_initializer().run(session=sess)

  print('c1 + c2 =', addc.eval(session=sess))

def Graph():
  graph= tf.Graph()
  with graph.as_default():
    value1 = tf.constant([1,2])
    value2 = tf.constant([3,4])
    mul = value1 * value2
    division = value1 / value2

  # put graph into session
  with tf.Session(graph = graph) as mySess:
    tf.global_variables_initializer().run()
    print('v1 * v2 =', mySess.run(mul)) # [3 8]
    # or
    print('v1 * v2 =', mul.eval())

## Save memory
def Placeholder():
  graph= tf.Graph()
  with graph.as_default():
    value1 = tf.placeholder(dtype=tf.int64)
    value2 = tf.Variable([3,4], dtype=tf.int64)
    mul = value1 * value2

  # put graph into session
  with tf.Session(graph = graph) as mySess:
    tf.global_variables_initializer().run()
    value = load_from_remote()
    for partialValue in load_partial(value, 2):  # load 2 values
      runResult = mySess.run(mul, feed_dict={value1: partialValue})
      evalResult = mul.eval(feed_dict={value1: partialValue})  # fit value to placeholder
      print("value1 * value2 =", evalResult)

def load_from_remote():
	return [-x for x in range(1000)]

def load_partial(value, step):
	index = 0
	while index < len(value):
		yield value[index:index + step]
		index += step
	return


if __name__ == '__main__':
  Addition()
  
  Constant()
  
  Graph()

  Placeholder()