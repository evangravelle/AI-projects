# DQN implemented for the OpenAI gym pong environment
# Written by Evan Gravelle
# 11/7/2016

# Maximize your score in the Atari 2600 game Pong. In this environment,
# the observation is an RGB image of the screen, which is an array of
# shape (210, 160, 3). Each action is repeatedly performed for a duration
# of k frames, where k is uniformly sampled from {2,3,4}.

# An episode ends once one player has 20 points.
# DQN paper trains for 10 million frames, with epsilon linearly annealed
# from 1 to 0.1 in first million frames, then held constant.

# TODO: solve diverging Q issue

# TODO: Possibly add regularization? L2 seems to be better than L1

# TODO: it seems like the random growth of other Q values outweighs the decay from gamma.

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
# import sys
import os.path
import pickle

# HYPERPARAMETERS
epsilon_initial = 1.0
epsilon_final = 0.1
epsilon_cutoff = 1000000
num_episodes = 1000  # per execution of script
max_num_timesteps = 5000
memory_cap = 10000  # One million should take up about 1GB of RAM
batch_size = 1
gamma = 0.99
# learning_rate = .00025  # assuming RMSProp is used
learning_rate = .00001  # .02 was too fast, and .0001 also made Q diverge
target_fix_time = 10000
save_variables_time = 50000
ep_range = 10
verbose = False


# INITIALIZATIONS
env = gym.make('Pong-v0')
# env.monitor.start('./tmp/pong-1', force=True)
checkpoint_filename = 'pong_scores/model.ckpt'
iteration_filename = 'pong_scores/iterations.txt'
score_filename = 'pong_scores/score.txt'
ep_filename = 'pong_scores/episodes.txt'
Q_filename = 'pong_scores/avg_max_Q.txt'
hold_out_filename = 'pong_scores/hold_out'
replay_filename = 'pong_scores/replay.p'
num_actions = env.action_space.n
num_rows = 210
reduced_rows = 164
num_cols = 160
num_chan = 3
input_size = reduced_rows * num_cols
replay_memory = [np.zeros((memory_cap, reduced_rows, num_cols, 4), dtype=np.int8),
                 np.zeros(memory_cap, dtype=np.uint8),
                 np.zeros(memory_cap, dtype=np.int8),
                 np.zeros((memory_cap, reduced_rows, num_cols, 4), dtype=np.int8)]
# print "size of replay_memory: ", sys.getsizeof(replay_memory)
not_terminal = np.ones(memory_cap, dtype=int)
np.set_printoptions(precision=4)


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
if os.path.isfile(replay_filename):
    with open(replay_filename, 'rb') as replay_file:
        replay_memory = pickle.load(replay_file)
    print 'Loaded replay_memory'

# Initialize epsilon, which linearly decreases then remains constant
if total_iter <= epsilon_cutoff:
    epsilon = (epsilon_final - epsilon_initial) * total_iter / epsilon_cutoff + 1.0
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
s = tf.placeholder(tf.float32, shape=[None, reduced_rows, num_cols, 4])  # 1st dim is batch size
a = tf.placeholder(tf.int32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# First layer is max pooling to reduce the image to (?, 82, 80, 4)
h_pool0 = -tf.nn.max_pool(-s, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print "h_pool0 = ", h_pool0

# Second layer is 16 8x8 convolutions followed by ReLU (?, 21, 20, 64)
W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 16], mean=0.0, stddev=tf.sqrt(2./64.)))
b_conv1 = tf.Variable(tf.constant(0.0, shape=[16]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(h_pool0, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
print "h_conv1 = ", h_conv1

# Third layer is 32 4x4 convolutions followed by ReLU (?, 11, 10, 128)
W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], mean=0.0, stddev=tf.sqrt(2./16.)))
b_conv2 = tf.Variable(tf.constant(0.0, shape=[32]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
print "h_conv2 = ", h_conv2

# Fourth layer is fully connected layer followed by ReLU, with arbitrary choice of 256 neurons
W_fc1 = tf.Variable(tf.truncated_normal([11 * 10 * 32, 256], mean=0.0, stddev=tf.sqrt(2.0/(11*10*32))))
b_fc1 = tf.Variable(tf.constant(0.0, shape=[256]))
h_conv2_flat = tf.reshape(h_conv2, [-1, 11 * 10 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
print "h_fc1 = ", h_fc1

# Fifth layer is output
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = tf.Variable(tf.truncated_normal([256, num_actions], mean=0.0, stddev=tf.sqrt(2.0/256.)))
b_fc2 = tf.Variable(tf.constant(0.0, shape=[num_actions]))
Q_vals = tf.matmul(h_fc1, W_fc2) + b_fc2
print "Q_vals = ", Q_vals

# Loss function is average (over mini-batch) mean squared error
loss = tf.reduce_mean((y - tf.matmul(Q_vals, tf.transpose(tf.one_hot(a, num_actions)))) ** 2, reduction_indices=[1])
# print "one_hot = ", tf.transpose(tf.one_hot(a, num_actions))

# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.95, epsilon=0.01).minimize(loss)
# train_step = tf.train.AdamOptimizer().minimize(loss)

# START SESSIONS
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess1 = tf.Session()
sess2 = tf.Session()
saver = tf.train.Saver()
if os.path.isfile(checkpoint_filename):
    saver.restore(sess1, checkpoint_filename)
    print 'Model restored to sess1 from', checkpoint_filename
    saver.restore(sess2, checkpoint_filename)
    print 'Model restored to sess2 from', checkpoint_filename
else:
    # with sess2.as_default():
    sess1.run(tf.initialize_all_variables())
    print 'Model initialized in sess1', checkpoint_filename
    saver.save(sess1, checkpoint_filename)
    print 'Model from sess1 saved in', checkpoint_filename
    # with sess1.as_default():
    saver.restore(sess2, checkpoint_filename)
    print 'Model restored to sess2 from', checkpoint_filename, '\n'


# LOAD OR CREATE HOLD OUT SET
if os.path.isfile(hold_out_filename):
    hold_out_set = np.load(hold_out_filename)
    hold_out_length = np.shape(hold_out_set)[0]
else:
    hold_out_set = np.zeros((max_num_timesteps, reduced_rows, num_cols, 4))
    hold_out_set[0, :, :, 0] = reduce_image(env.reset())
    action = env.action_space.sample()
    hold_out_set[0, :, :, 1] = reduce_image(env.step(action)[0])
    action = env.action_space.sample()
    hold_out_set[0, :, :, 2] = reduce_image(env.step(action)[0])
    action = env.action_space.sample()
    hold_out_set[0, :, :, 3] = reduce_image(env.step(action)[0])
    avg_max_Q = 0.
    for t in range(1, max_num_timesteps):
        action = env.action_space.sample()
        obs, reward, done = env.step(action)[:3]
        # env.render()
        obs_reduced = reduce_image(obs)
        hold_out_set[t, :, :, 0:3] = hold_out_set[t - 1, :, :, 1:4]
        hold_out_set[t, :, :, 3] = obs_reduced
        Q_vals_arr = sess1.run(Q_vals, feed_dict={s: hold_out_set[t, :, :, :].reshape(1, reduced_rows, num_cols, 4)})
        avg_max_Q += np.amax(Q_vals_arr, axis=1)
        if done:
            break
    hold_out_length = t + 1
    hold_out_set = hold_out_set[:hold_out_length, :, :, :]
    np.save(hold_out_filename, hold_out_set)
    if not os.path.isfile(Q_filename):
        avg_max_Q /= hold_out_length
        with open(Q_filename, 'w') as Q_file:
            Q_file.write(str(float(avg_max_Q)) + '\n')


# TRAINING LOOP
avg_score = 0.0
ep = start_ep
while ep < start_ep + num_episodes:
    print "episode ", ep
    # Assumes no reward is obtained in first 4 frames
    replay_ind = total_iter % memory_cap
    replay_memory[0][replay_ind, :, :, 0] = reduce_image(env.reset())
    action = env.action_space.sample()
    replay_memory[0][replay_ind, :, :, 1] = reduce_image(env.step(action)[0])
    action = env.action_space.sample()
    replay_memory[0][replay_ind, :, :, 2] = reduce_image(env.step(action)[0])
    action = env.action_space.sample()
    replay_memory[0][replay_ind, :, :, 3] = reduce_image(env.step(action)[0])
    obs, reward = env.step(action)[:2]
    obs_reduced = reduce_image(obs)
    # env.render()
    avg_score += reward

    for t in range(1, max_num_timesteps):

        current_batch_size = min([total_iter + 1, batch_size])
        # print "current_batch_size = ", current_batch_size
        current_replay_length = min([total_iter + 1, memory_cap])
        # print "current_replay_length = ", current_replay_length
        current_replays = random.sample(xrange(current_replay_length), current_batch_size)
        # print "current_replays = ", current_replays

        Q_vals_for_action = sess1.run(Q_vals, feed_dict={
          s: replay_memory[0][replay_ind, :, :, :].reshape(1, reduced_rows, num_cols, 4)})
        action = epsilon_greedy(epsilon, Q_vals_for_action)
        obs, reward, done = env.step(action)[:3]
        # env.render()
        avg_score += reward
        obs_reduced = reduce_image(obs)
        if verbose and t % 1 == 1:
            plt.imshow(obs_reduced, interpolation='nearest', cmap='Greys')
            plt.show()

        if done:
            not_terminal[replay_ind] = 0
        else:
            not_terminal[replay_ind] = 1

        replay_memory[1][replay_ind] = action
        replay_memory[2][replay_ind] = reward
        replay_memory[3][replay_ind, :, :, 0:3] = replay_memory[0][replay_ind, :, :, 1:4]
        replay_memory[3][replay_ind, :, :, 3] = obs_reduced

        # currently inefficient implementation, consider using partial_run (experimental)
        # Note, intermediate tensors are freed at the end of a sess.run()
        if verbose:
            prev_Q_vals_arr = sess1.run(Q_vals, feed_dict={
              s: replay_memory[0][current_replays, :, :, :].reshape(current_batch_size, reduced_rows, num_cols, 4)})
            Q_vals_arr = sess1.run(Q_vals, feed_dict={
              s: replay_memory[3][current_replays, :, :, :].reshape(current_batch_size, reduced_rows, num_cols, 4)})

        target_vals_arr = sess2.run(Q_vals, feed_dict={
          s: replay_memory[3][current_replays, :, :, :].reshape(current_batch_size, reduced_rows, num_cols, 4)})
        Q_max = np.amax(target_vals_arr, axis=1)
        r = replay_memory[2][current_replays]
        nt = not_terminal[current_replays]
        target = r + gamma * Q_max * nt

        train_step.run(feed_dict={s: replay_memory[0][current_replays, :, :, :].reshape(
                                    current_batch_size, reduced_rows, num_cols, 4),
                                  a: replay_memory[1][current_replays],
                                  y: target}, session=sess1)

        # print out stuff
        if verbose:
            ind = np.argmin(replay_memory[2][current_replays])
            # if a negative reward is received
            # if total_iter % int(target_fix_time/2.) == 1:
            # if float(replay_memory[2][current_replays[ind]]) < -.5:
            if total_iter % 1 == 0:
                prev_Q_vals_arr_after = sess1.run(Q_vals, feed_dict={
                  s: replay_memory[0][current_replays, :, :, :].reshape(current_batch_size, reduced_rows, num_cols, 4)})
                print 'total_iter = ', total_iter
                print 'reward = ', replay_memory[2][current_replays[ind]]
                print 'Q_vals_arr = ', Q_vals_arr[ind, :]
                print 'target_vals_arr = ', target_vals_arr[ind, :]
                # print 'Q_max = ', Q_max[ind]
                # print 'nt = ', nt[ind]
                print 'target = ', target[ind]
                print 'action = ', replay_memory[1][current_replays[ind]]
                print 'previous_Q_vals = ', prev_Q_vals_arr[ind, :]
                print 'Q_vals_delta =    ', prev_Q_vals_arr_after[ind, :] - prev_Q_vals_arr[ind, :], '\n'

        # After save_variables_time iterations, save variables and statistics
        if total_iter % save_variables_time == 0 and total_iter != 0:
            with open(replay_filename, 'w') as replay_file:
                pickle.dump(replay_memory, replay_file)
            with open(iteration_filename, 'w') as iter_file:
                iter_file.write(str(total_iter))

        # After target_fix_time iterations, fix new parameters for target
        if total_iter % target_fix_time == 0 and total_iter != 0:
            saver.save(sess1, checkpoint_filename)
            print 'Model from sess1 saved in', checkpoint_filename
            if os.path.isfile(checkpoint_filename):
                saver.restore(sess2, checkpoint_filename)
                print 'Model restored to sess2 from', checkpoint_filename
            avg_max_Q = 0.
            hold_out_vals_arr = sess1.run(Q_vals, feed_dict={s: hold_out_set.reshape(-1, reduced_rows, num_cols, 4)})
            avg_max_Q = np.sum(np.amax(hold_out_vals_arr, axis=1)) / hold_out_length
            with open(Q_filename, 'a') as Q_file:
                Q_file.write(str(avg_max_Q) + '\n')

        total_iter += 1
        old_replay_ind = replay_ind
        replay_ind = total_iter % memory_cap
        replay_memory[0][replay_ind, :, :, :] = replay_memory[3][old_replay_ind, :, :, :]
        if total_iter <= epsilon_cutoff:
            epsilon = (epsilon_final - epsilon_initial) * total_iter / epsilon_cutoff + 1.0
        if done:
            break

    if ep % ep_range == ep_range - 1:
        with open(ep_filename, 'w') as ep_file:
            ep_file.write(str(ep + 1))
        with open(score_filename, 'a') as score_file:
            score_file.write(str(avg_score / ep_range) + '\n')
        avg_score = 0.0
    ep += 1

sess1.close()
sess2.close()
# env.monitor.close()
