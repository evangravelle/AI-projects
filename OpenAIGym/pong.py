# DQN
# Implemented for the OpenAI gym pong environment
# Written by Evan Gravelle
# 8/5/2016

# Maximize your score in the Atari 2600 game Pong. In this environment,
# the observation is an RGB image of the screen, which is an array of
# shape (210, 160, 3) Each action is repeatedly performed for a duration
# of kk frames, where kk is uniformly sampled from {2,3,4}

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Initializations
env = gym.make('Pong-v0')
# env.monitor.start('./tmp/pong-1', force=True)
num_actions = env.action_space.n
num_rows = 210
num_cols = 160
num_chan = 3

# Parameters
epsilon = 0.1
epsilon_final = 0.1
num_episodes = 1
num_timesteps = 100


def reduce_image(_obs):
    new_obs = np.sum(_obs, 2) / (3. * 256.)
    new_obs[new_obs < .5] = 0
    new_obs[new_obs >= .5] = 1
    return new_obs


# Returns an action following an epsilon-greedy policy
def epsilon_greedy(_epsilon, _vals):
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = env.action_space.sample()
    return int(_action)


epsilon_coefficient = (epsilon - epsilon_final) ** (1. / num_episodes)
ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)

sess = tf.InteractiveSession()
# assume there is an observation "obs"
obs_reduced = reduce_image(obs)
x = -tf.nn.max_pool(-obs_reduced, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder(tf.float32, shape=[None, num_rows*num_cols/4.])  # 1st dim is batch size
y_ = tf.placeholder(tf.float32, shape=[None, num_actions])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean=0.0, stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
x_image = tf.reshape(x, [-1, num_rows/2., num_cols/2., 1])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean=0.0, stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully connected layer, with arbitrary choice of 1024 neurons
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Output layer with dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

start_time = datetime.datetime.now().time()


# Training loop
for ep in range(num_episodes):
    obs = env.reset()

    # Each episode
    for t in range(num_timesteps):

        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        rescaled_obs = reduce_image(obs)
        print rescaled_obs
        if done:
            break

    ep_length[ep] = t
    epsilon *= epsilon_coefficient

plt.imshow(rescaled_obs, cmap='Greys', interpolation='nearest')
plt.show()
# plt.hist(rescaled_obs.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# plt.show()
# env.monitor.close()
