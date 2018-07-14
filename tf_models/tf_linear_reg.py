'''
https://www.bilibili.com/video/av25528464/?p=5
'''
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Create 10000 random number, around y = 0.1x + 0.3
num_points = 1000
vectors_set = []
for i in range(num_points):
    x = np.random.normal(0.0, 0.55)  # mean = 0, std = 0.55
    y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x, y])


x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# plt.scatter(x_data, y_data, c='r')
# plt.show()

#  create values follow a uniform distribution in the range [minval, maxval)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data), name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss, name='train')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Print the init value of W and b
print("W=", sess.run(W), "b=", sess.run(b), 'loss=', sess.run(loss))

# Train 20 times
for step in range(20):
    sess.run(train)
    print("W=", sess.run(W), "b=", sess.run(b), 'loss=', sess.run(loss))

plt.scatter(x_data, y_data, c='r')
plt.plot(x_data, sess.run(W)*x_data + sess.run(b))
plt.show()
sess.close()