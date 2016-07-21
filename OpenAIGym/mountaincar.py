import gym
import numpy as np
env = gym.make('MountainCar-v0')


# FIX ELIGIBILITY TRACES! THEY ARENT CONTINUOUS


# There are 3 discrete actions, presumably accelerate left, right, or do nothing
# There are 2 observations, one in [-1.2, 0.6] and the other in [-0.07, 0.07]
# Likely x and xdot

# env.monitor.start('./tmp/cartpole-experiment-1', force=True)
num_actions = 3
dim = env.observation_space.high.size
xbar = np.zeros((2, dim))
xbar[0, :] = env.observation_space.low
xbar[1, :] = env.observation_space.high
num_row = 11
num_col = 11
width = 1. / (num_row - 1.)
height = 1. / (num_col - 1.)
sigma = width / 2.
den = 2 * sigma ** sigma
state = np.random.random(2)

# Learning Parameters
epsilon = 0.1
Lambda = 0.9
alpha = 0.5
gamma = 0.5

# The state-action value function is of the form V(s) = theta.T * phi(s).
# Initializing theta to zeros is appropriately optimistic
theta = np.zeros((num_row * num_col, num_actions))
e = np.zeros((num_row * num_col, num_actions))
c = np.zeros((num_row * num_col, dim))
for i in range(num_row):
    for j in range(num_col):
        k = i*num_col + j
        c[k, :] = (i * height, j * width)


# Various functions
def normalize_obs(_s):
    _y = np.zeros(dim)
    for _i in range(len(_y)):
        _y[i] = (_s[_i] - xbar[0, _i])/(xbar[1, _i] - xbar[0, _i])
    return _y


def phi(_state, _k):
    return np.exp(-np.linalg.norm(_state - c[_k]) / den)


def epsilon_greedy(_epsilon, _vals):
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = np.floor(_rand * _vals.size)
    return _action


def action_values(_state, _theta):
    _shape = _theta.shape
    _val = np.zeros(_shape[1])
    for _i in range(_shape[1]):
        for _k in range(_shape[0]):
            _val[_i] += _theta[_k, _i] * phi(_state, _i)
    return _val


def action_value(_state, _action, _theta):
    _shape = _theta.shape
    _val = np.zeros(_shape[1])
    for _k in range(_shape[0]):
        _val[_action] += _theta[_k, _action] * phi(_state, _action)
    return _val


for ep in range(1):
    state = env.reset()
    vals = action_values(state, theta)
    action = epsilon_greedy(epsilon, vals)
    for t in range(200):
        env.render()
        new_state, reward, done, info = env.step(action)
        new_vals = action_values(new_state, theta)
        new_action = epsilon_greedy(epsilon, new_vals)
        print new_state, new_action, theta.shape, state, action
        delta = reward + gamma * action_value(new_state, new_action, theta) - action_value(state, action, theta)
        e[state, action] += 1
        for i in range(num_row * num_col):
            for k in range(num_actions):
                theta[i, k] += alpha * delta * e[i, k]
                e[i, k] *= gamma * Lambda
        state = new_state
        action = new_action
        if done:
            break

# env.monitor.close()