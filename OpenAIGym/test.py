import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('Pong-v0')


def reduce_image(_obs):
    new_obs = np.sum(_obs, 2) / (3. * 256.)
    new_obs[new_obs < .4] = 0
    new_obs[new_obs >= .4] = 1
    return new_obs[33:194, :]

env.reset()
for i in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())  # take a random action
    print np.shape(obs)
    obs_reduced = reduce_image(obs)
    # plt.imshow(obs_reduced, cmap='Greys', interpolation='nearest')
    # plt.show()

    if i % 50 == 0:
        plt.imshow(obs_reduced, cmap='Greys', interpolation='nearest')
        plt.show()