import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
 
np.random.seed(101)
tf.set_random_seed(101)

n_features = 10
n_dense_neurons = 3

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

plt.plot(x_data, y_label, '*')
# plt.show()

np.random.rand(2)  # start m, b from 2 random number
m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0
for x, y in zip(x_data, y_label):
    y_hat = m * x +b    # the predicted value
    error += (y - y_hat) ** 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_steps = 10
    for i in range(train_steps):
        sess.run(train)

    final_slope, final_intercept = sess.run([m,b])


x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot)
plt.show()