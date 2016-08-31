# DQN implemented for the OpenAI gym pong environment
# Written by Evan Gravelle
# 8/5/2016

# Maximize your score in the Atari 2600 game Pong. In this environment,
# the observation is an RGB image of the screen, which is an array of
# shape (210, 160, 3) Each action is repeatedly performed for a duration
# of k frames, where k is uniformly sampled from {2,3,4}

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import datetime

# Initializations
env = gym.make('Pong-v0')
# env.monitor.start('./tmp/pong-1', force=True)
num_actions = env.action_space.n
num_rows = 210
num_cols = 160
num_chan = 3
input_size = num_rows * num_cols / 4.
memory_cap = 1000
replay_memory = np.zeros(memory_cap, input_size + 1 + 1 + input_size)
is_terminal = np.zeros(memory_cap)
replay_count = 0
batch_size = 32
Lambda = 0.95

# Parameters
epsilon = 1
epsilon_final = 0.1
num_episodes = 1
num_timesteps = 10000


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

# x is training input, y_ is training output
x = tf.placeholder(tf.float32, shape=[None, num_rows*num_cols])  # 1st dim is batch size
y_ = tf.placeholder(tf.float32, shape=[None, num_actions])

# First layer is max pooling to reduce the image to 105x80
x_image = tf.reshape(x, [-1, num_rows, num_cols, 1])
h_pool0 = -tf.nn.max_pool(-x_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Second layer is 16 8x8 ReLU convolutions
W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], mean=0.0, stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(h_pool0, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)

# Third layer is 32 4x4 ReLU convolutions
W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], mean=0.0, stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1)

# Fourth layer is fully connected ReLU layer, with arbitrary choice of 256 neurons
W_fc1 = tf.Variable(tf.truncated_normal([input_size * 32, 256], mean=0.0, stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv2_flat = tf.reshape(h_conv2, [-1, input_size * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

# Fifth layer is output with dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = tf.Variable(tf.truncated_normal([256, num_actions], mean=0.0, stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_actions]))
Q_vals = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss = (y - Q_vals[1]) ** 2
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

start_time = datetime.datetime.now().time()


# Training loop
for ep in range(num_episodes):
    prev_obs = env.reset()

    # Take a random action at first time step
    action = env.action_space.sample()
    obs, reward = env.step(action)
    obs_reduced = reduce_image(obs)
    env.render()
    obs_diff = prev_obs_reduced - reduce_image(prev_obs)

    for t in range(1, num_timesteps):
        Q_vals = sess.run(Q_vals, feed_dict={x: obs_diff, y_: reward})
        prev_obs_reduced = obs_reduced[:]
        prev_obs_diff = obs_diff[:]

        action = epsilon_greedy(epsilon, Q_vals)
        obs, reward, done, info = env.step(action)
        obs_reduced = reduce_image(obs)
        env.render()
        obs_diff = obs_reduced - prev_obs_reduced
        replay_ind = t - 1 % memory_cap
        print obs_diff
        if done:
            is_terminal[replay_ind] = 1
        replay_memory[replay_ind, :] = [prev_obs_diff, action, reward, obs_diff]
        current_batch_size = min([t, batch_size])
        current_replay_max = min([t - 1, memory_cap])
        current_replays = random.sample(set(xrange(current_replay_max)), current_batch_size)
        y = np.zeros(current_batch_size)
        for j in current_replays:
            if is_terminal[j]:
                y[j] = replay_memory[j, input_size + 1]
            else:
                Q_vals = sess.run(Q_vals, feed_dict={x: obs_diff, y_: reward})
                y[j] = replay_memory[j, input_size + 1] + Lambda * max(Q_vals)

        if done:
            break
        prev_obs_reduced = obs_reduced[:]
        prev_obs_diff = obs_diff[:]

    ep_length[ep] = t
    epsilon *= epsilon_coefficient

end_time = datetime.datetime.now().time()
plt.imshow(obs_reduced, cmap='Greys', interpolation='nearest')
plt.show()
# plt.hist(rescaled_obs.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# plt.show()
# env.monitor.close()
