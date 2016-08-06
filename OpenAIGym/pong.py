# DQN
# Implemented for the OpenAI gym pong environment
# Written by Evan Gravelle
# 8/5/2016

import gym
import numpy as np
import matplotlib.pyplot as plt

# Initializations
# Observations are (210, 160, 3) arrays, the output pixels
# 6 actions, probably correspond to moving with certain velocities
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
num_timesteps = 200

epsilon_coefficient = (epsilon - epsilon_final) ** (1. / num_episodes)
ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)


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
    state = env.reset()

    # Each episode
    for t in range(num_timesteps):

        env.render()
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)

        if done:
            break

    ep_length[ep] = t
    epsilon *= epsilon_coefficient

# env.monitor.close()
