'''
    Implement House Price prediction
    Steps:
        Prepare data
        Inference
        Loss Calculation
        Optimize
    Tensors types:
        Constant
        Variable
        PlaceHolder: used to pass data into graph
'''
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# generage some house sizes between 1000 and 3500 (typical sq ft of house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low =20000, high =70000, size=num_house)

# plot generaged house and size
plt.plot(house_size, house_price, "bx")  # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
# plt.show()


def normalize(array):
    return (array - array.mean())/ array.std()

# define number of training samples: 70%
num_train_samples = int(math.floor(num_house * 0.7))

#define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define the test data
test_house_size = np.asarray(house_size[:num_train_samples])
test_price = np.asanyarray(house_price[:num_train_samples])

test_house_size_norm = normalize(train_house_size)
test_house_price_norm = normalize(train_price)

# Setup the tensorflow placeholder that get undated as we descend down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf