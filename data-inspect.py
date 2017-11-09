'''
code https://github.com/CreatCodeBuild/deep-learning-capstone/blob/master/load.py
Data: http://ufldl.stanford.edu/housenumbers/
'''
from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np

def reformat(samples, labels):
  	# (height, width, channel, image count) -> (image count, height, width, channel)
	samples = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)

	# labels -> one-hot encoding [1]->[0,1,0,0,0,0,0,0,0,0]
	# digit 0 represented as 10
	# [10]->[1,0,0,0,0,0,0,0,0,0]
	labels = np.array([x[0] for x in labels])	# slow code, whatever
	one_hot_labels = []
	for num in labels:
		one_hot = [0.0] * 10
		if num == 10:
			one_hot[0] = 1.0
		else:
			one_hot[num] = 1.0
		one_hot_labels.append(one_hot)
	labels = np.array(one_hot_labels).astype(np.float32)
	# linearly normalize the image value from 0 - 255 to -1.0 to 1.0
	return samples, labels

# to Gray-scale, RGB -> number
# map 0 ~ 255 to -1.0 to +1.0
def normalize(samples):
	rgb = np.add.reduce(samples, keepdims=True, axis=3)
	rgb = rgb / 3.0
	return rgb / 128.0 - 1.0

def inspect(dataset, labels, i):
	print(labels[i])
	plt.imshow(dataset[i])
	plt.show()
	
# Find distribution / proportion of each label
def distribution(labels, name):
	count = {}
	for label in labels:
		key = 0 if label[0] == 10 else label[0]
		if key in count:
			count[key] += 1
		else:
			count[key] = 1
	x = []
	y = []
	for k, v in count.items():
		print(k, v)
		x.append(k)
		y.append(v)
	# draw x, y
	objects = x
	y_pos = np.arange(len(objects))
	performance = y
	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Count')
	plt.title(name + ' Label Distribution')
	plt.show()


# Load data
train = load('data/housenumber/train_32x32.mat')
test = load('data/housenumber/test_32x32.mat')

# Get samples and labels
train_samples = train['X']
train_labels = train['y']
# distribution(train_labels, "Train Labels")    # before normalize

test_samples = test['X']
test_labels = test['y']
# distribution(test_labels, "Test Labels")    # before normalize

_train_samples, _train_labels = reformat(train_samples, train_labels)
_test_samples, _test_labels = reformat(test_samples, test_labels)
# _train_samples = normalize(_train_samples)
# _test_samples = normalize(_test_samples)
inspect(_train_samples, _train_labels, 1234)

