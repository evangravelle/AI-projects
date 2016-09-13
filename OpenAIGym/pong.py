# DQN implemented for the OpenAI gym pong environment
# Written by Evan Gravelle
# 8/5/2016

# Maximize your score in the Atari 2600 game Pong. In this environment,
# the observation is an RGB image of the screen, which is an array of
# shape (210, 160, 3) Each action is repeatedly performed for a duration
# of k frames, where k is uniformly sampled from {2,3,4}

# An episode ends once one player has 20 points
# DQN paper trains for 10 million frames, with epsilon linearly annealed
# from 1 to 0.1 in first million frames, then held constant

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
reduced_rows = 164
num_cols = 160
num_chan = 3
input_size = reduced_rows * num_cols
memory_cap = 1000
replay_memory = np.zeros((memory_cap, input_size + 1 + 1 + input_size))
not_terminal = np.ones(memory_cap, dtype=int)
replay_count = 0
gamma = 0.99

# Parameters
epsilon = 1
epsilon_final = 0.1
num_episodes = 50
num_timesteps = 2000
batch_size = 32


# Returns cropped BW image of play area
# 0 is black, 1 is white.
def reduce_image(_obs):
    new_obs = np.sum(_obs, 2) / (3. * 256.)
    new_obs[new_obs < .4] = 0
    new_obs[new_obs >= .4] = 1
    return new_obs[32:196, :]


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

# x is training input, y_ is training output, these must be fed in later
s = tf.placeholder(tf.float32, shape=[None, input_size])  # 1st dim is batch size
a = tf.placeholder(tf.int32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
# print "s = ", s
# print "a = ", a
# print "y = ", y

# First layer is max pooling to reduce the image to (?, 82, 80, 1)
s_image = tf.reshape(s, [-1, reduced_rows, num_cols, 1])
h_pool0 = -tf.nn.max_pool(-s_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# print "h_pool0 = ", h_pool0

# Second layer is 16 8x8 convolutions followed by ReLU (?, 21, 20, 16)
W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], mean=0.0, stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(h_pool0, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
# print "h_conv1 = ", h_conv1

# Third layer is 32 4x4 convolutions followed by ReLU (?, 11, 10, 32)
W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], mean=0.0, stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
# print "h_conv2 = ", h_conv2

# Fourth layer is fully connected layer followed by ReLU, with arbitrary choice of 256 neurons
W_fc1 = tf.Variable(tf.truncated_normal([11 * 10 * 32, 256], mean=0.0, stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv2_flat = tf.reshape(h_conv2, [-1, 11 * 10 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
# print "h_fc1 = ", h_fc1

# Fifth layer is output with dropout
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = tf.Variable(tf.truncated_normal([256, num_actions], mean=0.0, stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_actions]))
Q_vals = tf.matmul(h_fc1, W_fc2) + b_fc2
# print "Q_vals = ", Q_vals

# Loss function is average mean squared error over mini-batch
loss = tf.reduce_mean((y - tf.matmul(Q_vals, tf.transpose(tf.one_hot(a, num_actions)))) ** 2)
# print "one_hot = ", tf.transpose(tf.one_hot(num_actions, a))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
train_step = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.initialize_all_variables())

start_time = datetime.datetime.now().time()

# Training loop
for ep in range(num_episodes):
    prev_obs = env.reset()

    # Take a random action at first time step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    obs_reduced = reduce_image(obs)
    # env.render()
    obs_diff = obs_reduced - reduce_image(prev_obs)

    for t in range(1, num_timesteps):
        if t % 100 == 0:
            print "t = ", t
        prev_obs_reduced = obs_reduced[:]
        prev_obs_diff = obs_diff[:]
        prev_Q_vals_arr = sess.run(Q_vals, feed_dict={s: prev_obs_diff.reshape((1, -1))})
        # print "Q_vals", prev_Q_vals_arr

        if t % 4 == 0:
            action = epsilon_greedy(epsilon, prev_Q_vals_arr)

        obs, reward, done, info = env.step(action)
        obs_reduced = reduce_image(obs)
        # env.render()
        obs_diff = obs_reduced - prev_obs_reduced
        replay_ind = (t - 1) % (memory_cap - 1)
        # print "replay_ind = ", replay_ind
        if False:
            plt.imshow(obs_reduced, cmap='Greys', interpolation='nearest')
            plt.show()

        if done:
            not_terminal[replay_ind] = 0
        replay_memory[replay_ind, :] = np.concatenate((prev_obs_diff.reshape(-1),
          (action,), (reward,), obs_diff.reshape(-1)))
        # print "replay_memory.size() = ", np.shape(replay_memory)
        current_batch_size = min([t, batch_size])
        # print "current_batch_size = ", current_batch_size
        current_replay_length = min([t, memory_cap])
        # print "current_replay_length = ", current_replay_length

        current_replays = random.sample(xrange(current_replay_length), current_batch_size)
        # print "current_replays = ", current_replays

        # currently inefficient implementation, consider using partial_run (experimental)
        # intermediate tensors are freed at the end of a sess.run()
        Q_vals_arr = sess.run(Q_vals, feed_dict={s: replay_memory[current_replays, input_size + 2:]})
        r = replay_memory[current_replays, input_size + 1]
        nt = not_terminal[current_replays]
        target = r + gamma * np.amax(Q_vals_arr, axis=1) * nt
        # print "target size = ", np.shape(target)

        train_step.run(feed_dict={s: replay_memory[current_replays, 0:input_size],
                                  a: replay_memory[current_replays, input_size],
                                  y: target})

        if done:
            break
        prev_action = action
        prev_obs_reduced = obs_reduced[:]
        prev_obs_diff = obs_diff[:]

    ep_length[ep] = t
    epsilon *= epsilon_coefficient
    # plt.imshow(obs, interpolation='nearest')
    im_str = "pong_scores/score%d" % ep
    plt.imsave(fname=im_str, arr=obs, format='png')

end_time = datetime.datetime.now().time()
# plt.hist(rescaled_obs.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# plt.show()
# env.monitor.close()
