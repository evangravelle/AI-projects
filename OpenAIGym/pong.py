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

epsilon_coefficient = (epsilon - epsilon_final) ** (1. / num_episodes)
ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)


def reduce_image(_obs):
    new_obs = np.sum(_obs, 2) / (3. * 256.)
    return new_obs


# Returns an action following an epsilon-greedy policy
def epsilon_greedy(_epsilon, _vals):
    _rand = np.random.random()
    if _rand < 1. - _epsilon:
        _action = _vals.argmax()
    else:
        _action = env.action_space.sample()
    return int(_action)


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
