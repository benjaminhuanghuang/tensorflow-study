import tensorflow as tf
import numpy as np


sess = tf.Session()
tens1 = tf.constant([[[1,2],[2,3]],[[3,4],[5,6]]])
print( sess.run(tens1))


x = tf.constant( np.random.rand(32).astype(np.float32))

np_array = np.array(([1,2,3],[3,4,5]))
tf_obj = tf.convert_to_tensor(np_array, dtype=tf.float32)
print(tf_obj)