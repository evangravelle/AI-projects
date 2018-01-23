""" 9x9 Go agent
Written by Evan Gravelle
01/22/2018"""

import gym
import numpy as np

np.set_printoptions(precision=2)

# Initializations
env = gym.make('Go9x9-v0')
observation = env.reset()

# Parameters
for _ in range(100):
    env.render()
    move = env.action_space.sample()
    row = move // 9
    col = move % 9
    # if not a legal move, resample
    if not observation[2, row, col]:
        move = env.action_space.sample()
        row = np.floor(move / 9)
        col = move % 9
    # I am X, opponent is 0
    # observation consists of 3 9x9 boards, 1st board is my pieces, 2nd board is their pieces, 3rd board is available spots
    # reward is 0 during game, +/- 1 for winning or losing
    # done is true when game ends
    # info contains a string which prints out the board
    observation, reward, done, info = env.step(0)

    if done:
        break
