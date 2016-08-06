# SARSA-lambda with Gaussian radial basis functions for action-value approximation
# Implemented for the OpenAI gym acrobot environment
# Written by Evan Gravelle
# 7/28/2016

# The acrobot system includes two joints and two links, where the joint
# between the two links is actuated. Initially, the links are hanging
# downwards, and the goal is to swing the end of the lower link up to
# a given height.

import gym
import numpy as np
import matplotlib.pyplot as plt

# Initializations
env = gym.make('Acrobot-v0')
env.monitor.start('./tmp/acrobot-1', force=True)
num_actions = env.action_space.n
dim = env.observation_space.high.size

# Parameters
discrt = 4
num_rbf = discrt * np.ones(dim).astype(int)
width = 1. / (num_rbf - 1.)
rbf_sigma = width[0] / 2.
epsilon = 0.1
epsilon_final = 0.1
Lambda = 0.5
alpha = 0.012
gamma = 0.99
num_episodes = 1000
num_timesteps = 200

xbar = np.zeros((2, dim))
xbar[0, :] = env.observation_space.low
xbar[1, :] = env.observation_space.high
num_ind = np.prod(num_rbf)
activations = np.zeros(num_ind)
new_activations = np.zeros(num_ind)
theta = np.zeros((num_ind, num_actions))
rbf_den = 2 * rbf_sigma ** 2
epsilon_coefficient = (epsilon - epsilon_final) ** (1. / num_episodes)
ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)


# Construct ndarray of rbf centers
c = np.zeros((num_ind, dim))
for i in range(num_ind):
    if i == 0:
        pad_num = dim
    else:
        pad_num = dim - int(np.log(i) / np.log(discrt)) - 1
    ind = np.base_repr(i, base=discrt, padding=pad_num)
    ind = np.asarray([float(j) for j in list(ind)])
    c[i, :] = width * ind


# Returns the state scaled between 0 and 1
def normalize_state(_s):
    _y = np.zeros(len(_s))
    for _i in range(len(_s)):
        _y[_i] = (_s[_i] - xbar[0, _i]) / (xbar[1, _i] - xbar[0, _i])
    return _y


# Returns an ndarray of radial basis function activations
def phi(_state):
    _phi = np.zeros(num_ind)
    for _k in range(num_ind):
        _phi[_k] = np.exp(-np.linalg.norm(_state - c[_k, :]) ** 2 / rbf_den)
    return _phi


# Returns an action following an epsilon-greedy policy
def epsilon_greedy(_epsilon, _vals):
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = env.action_space.sample()
    return int(_action)


# Returns the value of each action at some state
def action_values(_activations, _theta):
    _val = np.dot(_theta.T, _activations)
    return _val


# Returns the value of an action at some state
def action_value(_activations, _action, _theta):
    _val = np.dot(_theta[:, _action], _activations)
    return _val


# SARSA loop
for ep in range(num_episodes):

    e = np.zeros((num_ind, num_actions))
    state = normalize_state(env.reset())
    activations = phi(state)
    # print "activations = ", np.reshape(activations.ravel(order='F'), (num_rows, num_cols))
    vals = action_values(activations, theta)
    action = epsilon_greedy(epsilon, vals)

    # Each episode
    for t in range(num_timesteps):

        env.render()
        new_state, reward, done, info = env.step(action)
        new_state = normalize_state(new_state)
        new_activations = phi(new_state)
        new_vals = action_values(new_activations, theta)
        new_action = epsilon_greedy(epsilon, new_vals)
        Q = action_value(activations, action, theta)
        Q_new = action_value(new_activations, new_action, theta)
        if done:
            target = reward - Q
        else:
            target = reward + gamma * Q_new - Q
        # e[:, action] += activations  # accumulating traces
        e[:, action] = activations  # replacing traces

        for k in range(num_ind):
            for a in range(num_actions):
                theta[k, a] += alpha * target * e[k, a]
        e *= gamma * Lambda

        if t % 1 != 0:
            print "t = ", t
            print "new_state = ", new_state
            print "new_activations = ", np.reshape(new_activations.ravel(order='F'), (num_rows, num_cols))
            print "new_vals", new_vals
            print "Q = ", Q
            print "Q_new = ", Q_new
            print "action = ", action
            print "target = ", target
            print "e =", e
            print "theta = \n", np.reshape(theta.ravel(order='F'), (num_actions, num_rows, num_cols))
            print "---------------------------------------------------------------------------"

        state = new_state.copy()
        activations = new_activations.copy()
        action = new_action
        if done:
            break

    ep_length[ep] = t
    epsilon *= epsilon_coefficient


plt.close('all')
plt.figure(1)
plt.plot(ep_length)
plt.title('Episode Length')
plt.ylabel('Completion Time')
plt.xlabel('Episode')
plt.show()
env.monitor.close()
