# Homework 1 - Prototype selection for nearest neighbor
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

# mnist contains train, validation, and test objects
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

x_1, y_1 = mnist.train.next_batch(mnist.train.num_examples)
x_2, y_2 = mnist.train.next_batch(mnist.validation.num_examples)
train_set = np.concatenate((x_1, x_2))
train_labels = np.concatenate((y_1, y_2))
num_train, dim_input = np.shape(train_set)
num_test = mnist.test.num_examples
test_set, test_labels = mnist.test.next_batch(num_test)

# Parameters
k = 10
max_iterations = 1000

# k-means clustering
centroids_ind = random.sample(xrange(num_train), k)
centroids = train_set[centroids_ind, :]


def get_distance(_x1, _x2):
    return np.linalg.norm(_x1 - _x2)


def get_centroid(_x):
    return np.mean(_x, axis=1)


def get_nn(_x1, _x):
    _dist = np.inf
    for _i, _ in np.shape(_x):
        _new_dist = get_distance(_x1, _x[i, :])
        if _new_dist < _dist:
            _dist = _new_dist
            _ind = _i
    return _ind


prev_cost = np.inf
cluster = np.zeros(num_train)
for i in xrange(max_iterations):

    # Assign a cluster to each point
    cost = 0.
    for x_ind, x1 in enumerate(train_set):
        dist = np.inf
        for centroid_ind, centroid in enumerate(centroids):
            new_dist = get_distance(x1, centroid)
            if new_dist < dist:
                dist = new_dist
                cluster[x_ind] = centroid_ind
        cost += dist
    cost /= num_train

    # Find new centroids for each cluster
    centroids = np.zeros((k, dim_input))
    for j in range(k):
        ctr = 0.
        for x_ind, x1 in enumerate(train_set):
            if cluster[x_ind] == j:
                centroids[j, :] += x1
                ctr += 1.
        if ctr == 0:
            centroids[j, :] = train_set[random.sample(xrange(num_train))]
        else:
            centroids[j, :] /= ctr

    print 'cost = ', cost
    if i > 0 and abs((cost - prev_cost) / prev_cost) < 0.001:
        break

    prev_cost = cost

# Find the training example closest to the mean
prototypes = np.zeros((k, dim_input))
prototype_labels = np.zeros(k)
for j in range(k):
    dist = np.inf
    for x_ind, x1 in enumerate(x):
        if cluster[x_ind] == j:
            new_dist = get_distance(x1, centroids[j])
            if new_dist < dist:
                dist = new_dist
                prototypes[j, :] = x1
                prototype_labels[j] = train_labels[x_ind]

# Test performance of this set of prototypes, compared to random selection of prototypes
rand_inds = random.sample(xrange(num_train), k)
rand_prototypes = train_set[rand_inds, :]
nns = np.zeros(k)
rand_nns = np.zeros(k)
for x_ind, x1 in enumerate(train_set):
    nns[x_ind] = get_nn(x1, prototypes)
    rand_nns[x_ind] = get_nn(x1, rand_prototypes)
error_rate = np.sum(np.equal(test_labels, prototype_labels[nns])) / num_train
error_rate_rand = np.sum(np.equal(test_labels, prototype_labels[rand_nns])) / num_train

print 'error_rate = ', error_rate
print 'error_rate_rand = ', error_rate_rand
