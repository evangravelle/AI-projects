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

# TODO: Possibly add regularization? L2 seems to be better than L1

# TODO: tune learning rate based on observed effects when full minibatch is used

# TODO: current issue, when gamma = 0.99, the target grows because Q_max grows,
# TODO: it seems like the random growth of other Q values outweighs the decay from gamma.
# TODO: Maybe I enforce that the Q values of other actions don't change in the loss function?

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import datetime
import sys
import os.path
import time
import pickle

# HYPERPARAMETERS
epsilon_initial = 1.0
epsilon_final = 0.1
eps_cutoff = 1000000
num_epochs = 100  # 100 episodes per epoch
num_episodes = 500  # per execution of script
num_timesteps = 2000
memory_cap = 100000  # One million should take up about 1GB of RAM
batch_size = 32
gamma = 0.99
learning_rate = 1e-4


# INITIALIZATIONS
env = gym.make('Pong-v0')
# env.monitor.start('./tmp/pong-1', force=True)
checkpoint_filename = 'pong_scores/model.ckpt'
iteration_filename = 'pong_scores/iterations.txt'
score_filename = 'pong_scores/score.txt'
ep_filename = 'pong_scores/episodes.txt'
Q_filename = 'pong_scores/Q_val.txt'
epoch_filename = 'pong_scores/epochs.txt'
hold_out_filename = 'pong_scores/hold_out'
replay_filename = 'pong_scores/replay.p'
num_actions = env.action_space.n
num_rows = 210
reduced_rows = 164
num_cols = 160
num_chan = 3
input_size = reduced_rows * num_cols
replay_memory = [np.zeros((memory_cap, input_size), dtype=bool),
                 np.zeros(memory_cap), np.zeros(memory_cap),
                 np.zeros((memory_cap, input_size), dtype=bool)]
# print "size of replay_memory: ", sys.getsizeof(replay_memory)
not_terminal = np.ones(memory_cap, dtype=int)
replay_count = 0
ep_length = np.zeros(10000)
avg_Q = np.zeros(num_epochs)
np.set_printoptions(precision=2)


# LOAD SAVED FILES, IF THEY EXIST
if os.path.isfile(iteration_filename):
    with open(iteration_filename) as iter_file:
        total_iter = int(iter_file.read())
    print 'Loaded total_iter = ', total_iter
else:
    total_iter = 0
if os.path.isfile(ep_filename):
    with open(ep_filename) as ep_file:
        start_ep = int(ep_file.read())
    print 'Loaded start_ep = ', start_ep
else:
    start_ep = 0
if os.path.isfile(epoch_filename):
    with open(epoch_filename) as epoch_file:
        epoch = int(epoch_file.read())
    print 'Loaded epoch = ', epoch
else:
    epoch = 0
if os.path.isfile(replay_filename):
    with open(replay_filename, 'rb') as replay_file:
        replay_memory = pickle.load(replay_file)
    print 'Loaded replay_memory = ', np.shape(replay_memory)

# Initialize epsilon, which linearly decreases then remains constant
if total_iter <= eps_cutoff:
    epsilon = (epsilon_final - epsilon_initial) * total_iter / eps_cutoff + 1.0
else:
    epsilon = epsilon_final


# FUNCTIONS
# Returns cropped BW image of play area, 0 is black, 1 is white.
def reduce_image(_obs):
    # Returns cropped BW image of play area, 0 is black, 1 is white.
    new_obs = np.sum(_obs, 2) / (3. * 256.)
    new_obs[new_obs < .4] = 0
    new_obs[new_obs >= .4] = 1
    return new_obs[32:196, :]


# Returns an action following an epsilon-greedy policy
def epsilon_greedy(_epsilon, _vals):
    assert 0.0 <= _epsilon <= 1.0, 'Epsilon is out of bounds'
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = env.action_space.sample()
    return int(_action)


# INITIALIZE DEEP Q-NETWORK
# He et al. recommend, for CNN with ReLUs, random Gaussian weights with zero mean and
# std = sqrt(2.0/n), where n is number of input nodes
s = tf.placeholder(tf.float32, shape=[None, input_size])  # 1st dim is batch size
a = tf.placeholder(tf.int32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# First layer is max pooling to reduce the image to (?, 82, 80, 1)
s_image = tf.reshape(s, [-1, reduced_rows, num_cols, 1])
h_pool0 = -tf.nn.max_pool(-s_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# print "h_pool0 = ", h_pool0

# Second layer is 16 8x8 convolutions followed by ReLU (?, 21, 20, 16)
W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], mean=0.0, stddev=tf.sqrt(2.0/64.)))
b_conv1 = tf.Variable(tf.constant(0.0, shape=[16]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(h_pool0, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
# print "h_conv1 = ", h_conv1

# Third layer is 32 4x4 convolutions followed by ReLU (?, 11, 10, 32)
W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], mean=0.0, stddev=tf.sqrt(2.0/16.)))
b_conv2 = tf.Variable(tf.constant(0.0, shape=[32]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
# print "h_conv2 = ", h_conv2

# Fourth layer is fully connected layer followed by ReLU, with arbitrary choice of 256 neurons
W_fc1 = tf.Variable(tf.truncated_normal([11 * 10 * 32, 256], mean=0.0, stddev=tf.sqrt(2.0/3520.)))
b_fc1 = tf.Variable(tf.constant(0.0, shape=[256]))
h_conv2_flat = tf.reshape(h_conv2, [-1, 11 * 10 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
# print "h_fc1 = ", h_fc1

# Fifth layer is output
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = tf.Variable(tf.truncated_normal([256, num_actions], mean=0.0, stddev=tf.sqrt(2.0/256.)))
b_fc2 = tf.Variable(tf.constant(0.0, shape=[num_actions]))
Q_vals = tf.matmul(h_fc1, W_fc2) + b_fc2
# print "Q_vals = ", Q_vals

# Loss function is average (over mini-batch) mean squared error
loss = tf.reduce_mean((y - tf.matmul(Q_vals, tf.transpose(tf.one_hot(a, num_actions)))) ** 2, reduction_indices=[1])
# print "one_hot = ", tf.transpose(tf.one_hot(a, num_actions))

# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
# train_step = tf.train.AdamOptimizer().minimize(loss)

# START SESSIONS
# start_time = datetime.datetime.now().time()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with sess1.as_default():
    sess1.run(tf.initialize_all_variables())
    saver1 = tf.train.Saver()
with sess2.as_default():
    sess2.run(tf.initialize_all_variables())
    saver2 = tf.train.Saver()
if os.path.isfile(checkpoint_filename):
    saver1.restore(sess1, checkpoint_filename)
    print 'Model restored from %s' % checkpoint_filename


# LOAD OR CREATE HOLD OUT SET
if os.path.isfile(hold_out_filename):
    hold_out_set = np.load(hold_out_filename)
    hold_out_length, _ = np.shape(hold_out_set)
else:
    hold_out_set = np.zeros((num_timesteps, input_size))
    prev_obs = env.reset()
    prev_obs_reduced = reduce_image(prev_obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # env.render()
    obs_reduced = reduce_image(obs)
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
        Q_vals_arr = sess1.run(Q_vals, feed_dict={s: hold_out_set[t, :].reshape(1, -1)})
        avg_Q[epoch] += np.amax(Q_vals_arr, axis=1)
        if done:
            break
    hold_out_length = t + 1
    hold_out_set = hold_out_set[:hold_out_length, :]
    np.save(hold_out_filename, hold_out_set)
    if epoch == 0:
        avg_Q[epoch] /= hold_out_length
        with open(Q_filename, 'a') as Q_file:
            Q_file.write(str(avg_Q[epoch]) + '\n')
        with open(epoch_filename, 'w') as epoch_file:
            epoch_file.write(str(epoch))
        epoch += 1

# Save initial variables to file, and load checkpoint in sess2
if total_iter == 0:
    save_path = saver1.save(sess1, checkpoint_filename)
    print "Model saved in file: %s" % save_path
    if os.path.isfile(checkpoint_filename):
        saver2.restore(sess2, checkpoint_filename)
        print 'Parameters copied from %s' % checkpoint_filename
    with open(iteration_filename, 'w') as iter_file:
        iter_file.write(str(total_iter))
    with open(ep_filename, 'w') as ep_file:
        ep_file.write(str(0))


# TRAINING LOOP
avg_score = 0.0
ep = start_ep
while ep < start_ep + num_episodes:
    print "episode ", ep
    prev_obs = env.reset()
    prev_obs_reduced = reduce_image(prev_obs)
    # Take a random action at first time step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # env.render()
    avg_score += reward
    obs_reduced = reduce_image(obs)
    obs_diff = obs_reduced - reduce_image(prev_obs)

    for t in range(1, num_timesteps):
        prev_obs_reduced = obs_reduced[:]
        prev_obs_diff = obs_diff[:]
        prev_Q_vals_toadd = sess1.run(Q_vals, feed_dict={s: prev_obs_diff.reshape((1, -1))})
        # print "previous_Q_vals = ", prev_Q_vals_arr
        action = epsilon_greedy(epsilon, prev_Q_vals_toadd)
        obs, reward, done, info = env.step(action)
        # env.render()
        avg_score += reward
        obs_reduced = reduce_image(obs)
        obs_diff = obs_reduced - prev_obs_reduced
        replay_ind = total_iter % memory_cap
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
        # Note, intermediate tensors are freed at the end of a sess.run()
        prev_Q_vals_arr = sess1.run(Q_vals, feed_dict={
          s: replay_memory[0][current_replays, :].reshape(current_batch_size, -1)})

        Q_vals_arr = sess2.run(Q_vals, feed_dict={
          s: replay_memory[3][current_replays, :].reshape(current_batch_size, -1)})
        r = replay_memory[2][current_replays]
        Q_max = np.amax(Q_vals_arr, axis=1)
        nt = not_terminal[current_replays]
        target = r + gamma * Q_max * nt

        train_step.run(feed_dict={s: replay_memory[0][current_replays, :].reshape(current_batch_size, -1),
                                  a: replay_memory[1][current_replays],
                                  y: target}, session=sess1)

        prev_Q_vals_arr_after = sess1.run(Q_vals, feed_dict={
          s: replay_memory[0][current_replays, :].reshape(current_batch_size, -1)})

        # print out stuff
        print 'reward = ', replay_memory[2][current_replays]
        print 'previous_Q_vals = ', prev_Q_vals_arr
        print 'Q_max = ', Q_max
        print 'nt = ', nt
        print 'target = ', target
        print 'action = ', replay_memory[1][current_replays]
        print 'Q_vals_delta =    ', prev_Q_vals_arr_after - prev_Q_vals_arr, '\n'
        if reward != 0.:
            plt.imshow(obs, interpolation='nearest')  # cmap='Greys'
            plt.show()
        if done:
            break
        total_iter += 1
        if total_iter <= eps_cutoff:
            epsilon = (epsilon_final - epsilon_initial) * total_iter / eps_cutoff + 1.0
        else:
            pass

    ep_length[ep] = t

    # Every 10 episodes, save variables and statistics, and fix new parameters for target
    if ep % 10 == 9:
        # im_str = "pong_scores/score%d" % ep
        # plt.imsave(fname=im_str, arr=obs, format='png')
        save_path = saver1.save(sess1, checkpoint_filename)
        print "Model saved in file: %s" % save_path
        if os.path.isfile(checkpoint_filename):
            saver2.restore(sess2, checkpoint_filename)
            print 'Parameters copied from %s to the target network' % checkpoint_filename
        with open(replay_filename, 'wb') as replay_file:
            pickle.dump(replay_memory, replay_file)
        with open(iteration_filename, 'w') as iter_file:
            iter_file.write(str(total_iter))
        with open(ep_filename, 'w') as ep_file:
            ep_file.write(str(ep+1))
        with open(score_filename, 'a') as score_file:
            score_file.write(str(avg_score/10.) + '\n')
        # TODO: feed this in as a batch, for efficiency
        for state in hold_out_set:
            # print "state dimension = ", np.shape(state)
            Q_vals_arr = sess1.run(Q_vals, feed_dict={s: state.reshape(1, -1)})
            avg_Q[epoch] += np.amax(Q_vals_arr.reshape(-1))
        avg_Q[epoch] /= hold_out_length
        with open(Q_filename, 'a') as Q_file:
            Q_file.write(str(avg_Q[epoch]) + '\n')
        with open(epoch_filename, 'w') as epoch_file:
            epoch_file.write(str(epoch))
        avg_score = 0.0
        epoch += 1

    ep += 1

# end_time = datetime.datetime.now().time()
sess1.close()
sess2.close()
# env.monitor.close()
