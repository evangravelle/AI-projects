import gym
import numpy as np
import matplotlib.pyplot as plt
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
Lambda = 0.5
alpha = 0.2
gamma = 0.95

# The state-action value function is of the form V(s) = theta.T * phi(s).
# Initializing theta to zeros is appropriately optimistic
theta = np.zeros((num_row * num_col, num_actions))
c = np.zeros((num_row * num_col, dim))


for i in range(num_row):
    for j in range(num_col):
        k = i*num_col + j
        c[k, :] = (i * height, j * width)


# Remaps the state to between 0 and 1
def normalize_state(_s):
    _y = np.zeros(len(_s))
    for _i in range(len(_s)):
        _y[_i] = (_s[_i] - xbar[0, _i]) / (xbar[1, _i] - xbar[0, _i])
    return _y


# outputs vector phi
def phi(_state):
    _phi = np.zeros(num_row * num_col)
    for _k in range(num_row * num_col):
        _phi[_k] = np.exp(-np.linalg.norm(_state - c[_k, :]) ** 2 / den)
    return _phi


def epsilon_greedy(_epsilon, _vals):
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = np.floor(_rand * _vals.size)
    return int(_action)


def action_values(_activations, _theta):
    _val = np.dot(_theta.T, _activations)
    return _val


def action_value(_activations, _action, _theta):
    _val = np.dot(_theta[:, _action], _activations)
    return _val


for ep in range(100):

    e = np.zeros((num_row * num_col, num_actions))
    state = normalize_state(env.reset())
    activations = phi(state)
    # print "activations = ", np.reshape(activations.ravel(order='F'), (num_row, num_col))
    vals = action_values(activations, theta)
    action = epsilon_greedy(epsilon, vals)
    for t in range(2000):
        env.render()
        new_state, reward, done, info = env.step(action)
        new_state = normalize_state(new_state)
        new_activations = phi(new_state)
        new_vals = action_values(new_activations, theta)
        new_action = epsilon_greedy(epsilon, new_vals)
        Q2 = action_value(new_activations, new_action, theta)
        Q1 = action_value(activations, action, theta)
        delta = (reward + gamma * action_value(new_activations, new_action, theta) -
                 action_value(activations, action, theta))

        for k in range(num_row * num_col):
            # e[:, action] += activations  # accumulating traces
            e[:, action] = activations  # replacing traces
            for a in range(num_actions):
                theta[k, a] += alpha * delta * e[k, a]
                e[k, a] *= gamma * Lambda

        if t % 1 != 0:
            print "t = ", t
            print "new_state = ", new_state
            print "new_activations = ", np.reshape(new_activations.ravel(order='F'), (num_row, num_col))
            print "Q_current = ", Q1
            print "Q_future = ", Q2
            print "action = ", action
            print "delta = ", delta
            print "e =", e
            print "theta = \n", np.reshape(theta.ravel(order='F'), (num_actions, num_row, num_col))
            print "---------------------------------------------------------------------------"

        # print "state = ", state
        state = new_state.copy()
        activations = new_activations.copy()
        action = new_action
        if done:
            break

value_left = np.zeros(num_row * num_col)
value_nothing = np.zeros(num_row * num_col)
value_right = np.zeros(num_row * num_col)

for h in range(num_row * num_col):
    current_activations = phi(c[h, :])
    value_left[h] += action_value(current_activations, 0, theta)
    value_nothing[h] += action_value(current_activations, 1, theta)
    value_right[h] += action_value(current_activations, 2, theta)

# print np.reshape(current_activations.ravel(order='F'), (num_row, num_col))

plt.close('all')
fig, axes = plt.subplots(ncols=3, sharey=True)
plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')
im = axes[0].imshow(value_left.reshape((num_row, num_col)), cmap='hot')
axes[0].set_title('Action = left')
axes[0].set_ylabel('Position')
axes[0].set_xlabel('Velocity')
im = axes[1].imshow(value_nothing.reshape((num_row, num_col)), cmap='hot')
axes[1].set_title('Action = nothing')
im = axes[2].imshow(value_right.reshape((num_row, num_col)), cmap='hot')
axes[2].set_title('Action = right')
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
plt.axis([0, 1, 0, 1])
plt.show()

# env.monitor.close()
