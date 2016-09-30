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

# Can try using epsilon = 0.05 at each epoch, as a better indicator of learning

# CURRENT ISSUE: when gamma = 0.99, the target grows because Q_max grows, it seems
# like the random growth of other Q values outweighs the decay from gamma
# Maybe I enforce that the Q values of other actions don't change in the loss function?

# Another idea: make the input "white", zero mean and unit variance

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import datetime
import sys
import os.path
import time

# Initializations
env = gym.make('Pong-v0')
# env.monitor.start('./tmp/pong-1', force=True)
checkpoint_filename = 'pong_scores/model.ckpt'
iteration_filename = 'pong_scores/iterations.txt'
score_filename = 'pong_scores/score.txt'
ep_filename = 'pong_scores/episodes.txt'
Q_filename = 'pong_scores/Q_val.txt'
epoch_filename = 'pong_scores/epochs.txt'
hold_out_filename = 'pong_scores/hold_out.txt'
num_actions = env.action_space.n
num_rows = 210
reduced_rows = 164
num_cols = 160
num_chan = 3
input_size = reduced_rows * num_cols
memory_cap = 500000  # One million should take up about 1GB of RAM
replay_memory = [np.zeros((memory_cap, input_size), dtype=bool),
                 np.zeros(memory_cap), np.zeros(memory_cap),
                 np.zeros((memory_cap, input_size), dtype=bool)]
# print "size of replay_memory: ", sys.getsizeof(replay_memory)
not_terminal = np.ones(memory_cap, dtype=int)
replay_count = 0

# Read saved files, if they exist
if os.path.isfile(iteration_filename):
    with open(iteration_filename) as iter_file:
        total_iter = int(iter_file.read())
else:
    total_iter = 0
if os.path.isfile(ep_filename):
    with open(ep_filename) as ep_file:
        start_ep = int(ep_file.read())
else:
    start_ep = 0
if os.path.isfile(epoch_filename):
    with open(epoch_filename) as epoch_file:
        total_iter = int(epoch_file.read())
else:
    epoch = 0
# print 'total_iter = ', total_iter

# Parameters
epsilon_initial = 1.0
epsilon_final = 0.1
eps_cutoff = 1000000
num_epochs = 100  # 100 episodes per epoch
num_episodes = 100  # per execution of script
num_timesteps = 2000
batch_size = 32
gamma = 0.9
learning_rate = 1e-4

avg_Q = np.zeros(num_epochs)
if total_iter <= eps_cutoff:
    epsilon = (epsilon_final - epsilon_initial) * total_iter / eps_cutoff + 1.0
else:
    epsilon = epsilon_final


# Returns cropped BW image of play area
# 0 is black, 1 is white.
def reduce_image(_obs):
    new_obs = np.sum(_obs, 2) / (3. * 256.)
    new_obs[new_obs < .4] = 0
    new_obs[new_obs >= .4] = 1
    return new_obs[32:196, :]


# Returns an action following an epsilon-greedy policy
def epsilon_greedy(_epsilon, _vals):
    assert 0.0 <= _epsilon <= 1.0, "Epsilon is out of bounds"
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = env.action_space.sample()
    return int(_action)


# epsilon_coefficient = (epsilon - epsilon_final) ** (1. / num_episodes)
ep_length = np.zeros(10000)
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

# Fifth layer is output (with dropout?)
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = tf.Variable(tf.truncated_normal([256, num_actions], mean=0.0, stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_actions]))
Q_vals = tf.matmul(h_fc1, W_fc2) + b_fc2
# print "Q_vals = ", Q_vals

# Loss function is average mean squared error over mini-batch
loss = tf.reduce_mean((y - tf.matmul(Q_vals, tf.transpose(tf.one_hot(a, num_actions)))) ** 2)
# print "one_hot = ", tf.transpose(tf.one_hot(a, num_actions))

# train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(-loss)
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Start session
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
if os.path.isfile(checkpoint_filename):
    saver.restore(sess, checkpoint_filename)
    print 'Model restored from %s' % checkpoint_filename
start_time = datetime.datetime.now().time()

# Create hold-out set for Q-value statistics
if os.path.isfile(hold_out_filename):
    hold_out_set = np.load(hold_out_filename)
    # load set here
else:
    hold_out_set = np.zeros((num_timesteps, input_size))
    prev_obs = env.reset()
    prev_obs_reduced = reduce_image(prev_obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    obs_reduced = reduce_image(obs)
    # env.render()
    obs_diff = obs_reduced - prev_obs_reduced
    hold_out_set[0, :] = obs_diff.reshape((1, -1))

    for t in range(1, num_timesteps):
        prev_obs_reduced = obs_reduced[:]
        prev_obs_diff = obs_diff[:]
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render()
        obs_reduced = reduce_image(obs)
        obs_diff = obs_reduced - prev_obs_reduced
        hold_out_set[t, :] = obs_diff.reshape((1, -1))
        Q_vals_arr = sess.run(Q_vals, feed_dict={s: hold_out_set[t, :].reshape(1, -1)})
        # print "Q_vals_arr = ", Q_vals_arr
        avg_Q[epoch] += max(Q_vals_arr.reshape(-1))
        if done:
            break
    hold_out_length = t + 1
    hold_out_set = hold_out_set[:hold_out_length, :]
    np.save(hold_out_filename, hold_out_set)
    avg_Q[epoch] /= hold_out_length

# Training loop
avg_score = 0.
ep = start_ep
while ep < start_ep + num_episodes:
    print "episode ", ep
    prev_obs = env.reset()
    prev_obs_reduced = reduce_image(prev_obs)
    # Take a random action at first time step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if reward == 1:
        avg_score += 1
    obs_reduced = reduce_image(obs)
    # env.render()
    obs_diff = obs_reduced - reduce_image(prev_obs)

    for t in range(1, num_timesteps):
        prev_obs_reduced = obs_reduced[:]
        prev_obs_diff = obs_diff[:]
        # if t >= 20:
            # plt.imshow(prev_obs_diff.reshape(-1, num_cols), cmap='Greys', interpolation='nearest')
            # plt.show()
            # pool_layer = sess.run(h_pool0, feed_dict={s: prev_obs_diff.reshape((1, -1))})
            # plt.imshow(pool_layer.reshape(-1, num_cols/2), cmap='Greys', interpolation='nearest')
            # plt.show()
            # plt.pause(2.0)
        prev_Q_vals_arr = sess.run(Q_vals, feed_dict={s: prev_obs_diff.reshape((1, -1))})
        # print "previo_Q_vals = ", prev_Q_vals_arr

        # I think it does the action holding already!
        # if t % 4 == 0:
        action = epsilon_greedy(epsilon, prev_Q_vals_arr)

        obs, reward, done, info = env.step(action)
        if reward == 1:
            avg_score += 1
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
        replay_memory[0][replay_ind, :] = prev_obs_diff.reshape(-1)
        replay_memory[1][replay_ind] = action
        replay_memory[2][replay_ind] = reward
        replay_memory[3][replay_ind, :] = obs_diff.reshape(-1)

        # print "replay_memory.size() = ", np.shape(replay_memory)
        current_batch_size = min([t, batch_size])
        # print "current_batch_size = ", current_batch_size
        current_replay_length = min([t, memory_cap])
        # print "current_replay_length = ", current_replay_length

        current_replays = random.sample(xrange(current_replay_length), current_batch_size)
        # print "current_replays = ", current_replays

        # currently inefficient implementation, consider using partial_run (experimental)
        # intermediate tensors are freed at the end of a sess.run()
        prev_Q_vals_arr = sess.run(Q_vals, feed_dict={
          s: replay_memory[0][current_replays, :].reshape(current_batch_size, -1)})
        Q_vals_arr = sess.run(Q_vals, feed_dict={
          s: replay_memory[3][current_replays, :].reshape(current_batch_size, -1)})
        # print 't = ', t
        # print 'epsilon = ', epsilon

        r = replay_memory[2][current_replays]
        nt = not_terminal[current_replays]
        target = r + gamma * np.amax(Q_vals_arr, axis=1) * nt

        train_step.run(feed_dict={s: replay_memory[0][current_replays, :].reshape(current_batch_size, -1),
                                  a: replay_memory[1][current_replays],
                                  y: target})

        prev_Q_vals_arr_after = sess.run(Q_vals, feed_dict={
          s: replay_memory[0][current_replays, :].reshape(current_batch_size, -1)})

        # print out stuff
        # print 'action = ', replay_memory[1][current_replays]
        # print 'reward = ', replay_memory[2][current_replays]
        # print "previo_Q_vals = ", prev_Q_vals_arr
        # print 'Q_max = ', np.amax(Q_vals_arr, axis=1)
        # print 'nt = ', nt
        # print "target = ", target
        # print 'change_Q_vals = ', prev_Q_vals_arr_after - prev_Q_vals_arr, '\n'
        if done:
            break
        total_iter += 1
        if total_iter <= eps_cutoff:
            epsilon = (epsilon_final - epsilon_initial) * total_iter / eps_cutoff + 1.0
        else:
            pass

    ep_length[ep] = t
    # print "epsilon = ", epsilon

    if ep % 10 == 9:
        # im_str = "pong_scores/score%d" % ep
        # plt.imsave(fname=im_str, arr=obs, format='png')
        save_path = saver.save(sess, checkpoint_filename)
        print "Model saved in file: %s" % save_path
        with open(iteration_filename, 'w') as iter_file:
            iter_file.write(str(total_iter))
        with open(ep_filename, 'w') as ep_file:
            ep_file.write(str(ep+1))
        with open(score_filename, 'a') as score_file:
            score_file.write(str(avg_score/10.) + '\n')
        avg_score = 0.

    # Every 10 episodes, record average max Q value at each state in hold out set
    # feed this in as a batch, for efficiency
    if ep % 10 == 9:
        for state in hold_out_set:
            # print "state dimension = ", np.shape(state)
            Q_vals_arr = sess.run(Q_vals, feed_dict={s: state.reshape(1, -1)})
            avg_Q[epoch] += max(Q_vals_arr.reshape(-1))
        avg_Q[epoch] /= hold_out_length
        with open(Q_filename, 'a') as Q_file:
            Q_file.write(str(avg_Q[epoch]) + '\n')
        epoch += 1

    ep += 1

end_time = datetime.datetime.now().time()
sess.close()
# plt.hist(rescaled_obs.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# plt.show()
# env.monitor.close()
