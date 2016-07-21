import gym
import numpy as np
env = gym.make('MountainCar-v0')

# plot the value function at certain time intervals!
# try using replacing or accumulating traces, Sutton says replacing are better

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
den = 2 * sigma ** 2
state = np.random.random(2)
activations = np.zeros(num_row * num_col)
new_activations = np.zeros(num_row * num_col)
np.set_printoptions(precision=2)

# Learning Parameters
epsilon = 0.1
Lambda = 0.9
alpha = 0.5
gamma = 0.9

# The state-action value function is of the form V(s) = theta.T * phi(s).
# Initializing theta to zeros is appropriately optimistic
theta = np.zeros((num_row * num_col, num_actions))
c = np.zeros((num_row * num_col, dim))
for i in range(num_row):
    for j in range(num_col):
        k = i*num_col + j
        c[k, :] = (i * height, j * width)


# Various functions
def normalize_state(_s):
    _y = np.zeros(len(_s))
    for _i in range(len(_s)):
        _y[_i] = (_s[_i] - xbar[0, _i]) / (xbar[1, _i] - xbar[0, _i])
    return _y


def phi(_state, _i):
    return np.exp(-np.linalg.norm(_state - c[_i, :]) ** 2 / den)


def epsilon_greedy(_epsilon, _vals):
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = np.floor(_rand * _vals.size)
    return int(_action)


def action_values(_activations, _theta):
    _shape = _theta.shape
    _val = np.zeros(_shape[1])
    for _i in range(_shape[1]):
        for _k in range(_shape[0]):
            _val[_i] += _theta[_k, _i] * _activations[_k]
    return _val


def action_value(_activations, _action, _theta):
    _shape = _theta.shape
    _val = 0
    for _k in range(_shape[0]):
        _val += _theta[_k, _action] * _activations[_k]
    return _val


for ep in range(20):

    e = np.zeros((num_row * num_col, num_actions))
    state = normalize_state(env.reset())

    for i in range(num_row * num_col):
        activations[i] = phi(state, i)

    # print "activations = ", np.reshape(activations.ravel(order='F'), (num_row, num_col))
    vals = action_values(activations, theta)
    action = epsilon_greedy(epsilon, vals)
    for t in range(1000):
        env.render()
        new_state, reward, done, info = env.step(action)
        new_state = normalize_state(new_state)

        for i in range(num_row * num_col):
            new_activations[i] = phi(new_state, i)

        new_vals = action_values(new_activations, theta)
        new_action = epsilon_greedy(epsilon, new_vals)
        Q2 = action_value(new_activations, new_action, theta)
        Q1 = action_value(activations, action, theta)
        delta = (reward + gamma * action_value(new_activations, new_action, theta) -
                 action_value(activations, action, theta))

        for i in range(num_row * num_col):
            # e[i, action] += activations[i]  # accumulating traces
            e[i, action] = activations[i]  # replacing traces
            for k in range(num_actions):
                theta[i, k] += alpha * delta * e[i, k]
                e[i, k] *= gamma * Lambda

        if t % 1 != 0:
            print "t = ", t
            # print "new_state = ", new_state
            # print "new_activations = ", np.reshape(new_activations.ravel(order='F'), (num_row, num_col))
            # print "Q_current = ", Q1
            # print "Q_future = ", Q2
            # print "action = ", action
            # print "delta = ", delta
            print "e =", e
            print "theta = \n", np.reshape(theta.ravel(order='F'), (num_actions, num_row, num_col))
            print "---------------------------------------------------------------------------"

        # print "state = ", state
        state = new_state.copy()
        activations = new_activations.copy()
        action = new_action
        if done:
            break

# env.monitor.close()
