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
    move = env.action_space.sample()  # 0 - 80 are moves (row-major), 81 is pass, 82 is resign

    # this code is ugly: define a bool which states if the move is legal or not
    if move >= 81:
        observation, reward, done, info = env.step(move)
        if done:
            break
    else:
        row = move // 9
        col = move % 9
        print(move, row, col)
        print(env.action_space)
        # if not a legal move, resample
        while not observation[2, row, col]:
            move = env.action_space.sample() - 1
            row = move // 9
            col = move % 9
            print(move, row, col)
        # I am X, opponent is 0
        # observation consists of 3 9x9 boards: my pieces, their pieces, and available spots
        # reward is 0 during game, +/- 1 for winning or losing
        # done is true when game ends
        # info contains a string which prints out the board
        observation, reward, done, info = env.step(move)

        if done:
            if reward == 1:
                print('VICTORY')
            elif reward == -1:
                print('DEFEAT')
            else:
                print('HUH?')
            break
