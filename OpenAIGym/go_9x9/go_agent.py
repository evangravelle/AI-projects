""" 9x9 Go agent
Written by Evan Gravelle
01/22/2018"""

import gym
import keras
import numpy as np

column = 'ABCDEFGHJ'


def get_move(move):
    if move == 81:
        return 'PASS'
    elif move == 82:
        return 'RESIGN'
    else:
        return column[move % 9] + str((89 - move) // 9)


class model(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    # Initializations
    env = gym.make('Go9x9-v0')
    observation = env.reset()
    max_moves = 81 * 2

    env.render()
    for _ in range(max_moves):
        # 0 - 80 are moves (row-major), 81 is pass, 82 is resign
        available_moves = np.squeeze(np.nonzero(observation[2].flatten()))
        move = np.random.choice(available_moves)
        print(get_move(move))

        # I am X, opponent is 0
        # observation consists of 3 9x9 boards: my pieces, their pieces, and available spots
        # reward is 0 during game, +/- 1 for winning or losing
        # done is true when game ends
        # info contains a string which prints out the board
        observation, reward, done, info = env.step(move)
        env.render()

        if done:
            if reward == 1:
                print('VICTORY')
            elif reward == -1:
                print('DEFEAT')
            else:
                print('HUH?')
            break
