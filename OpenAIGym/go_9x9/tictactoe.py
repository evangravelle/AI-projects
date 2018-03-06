""" Tic Tac Toe
Written by Evan Gravelle
01/22/2018 """

import math
import numpy as np
np.set_printoptions(precision=2)


class Player(object):
    def __init__(self):
        self.player = 0
        self.players = 'xo'

    def current_player(self):
        return self.players[self.player]

    def update_player(self):
        self.player = (self.player + 1) % 2
        return self.current_player()


class TicTacToe(object):
    def __init__(self):
        self.state = list('         ')
        self.players = Player()
        self.turn = 0

    def __str__(self):
        return (self.board() + '\nTurn ' + str(self.turn + 1) + ', ' +
                self.players.current_player().upper() + "\'s turn\n")

    # Returns tuple of (is_terminal, reward)
    def check_state(self):
        threes = [self.state[0] + self.state[3] + self.state[6],
                  self.state[1] + self.state[4] + self.state[7],
                  self.state[2] + self.state[5] + self.state[8],
                  self.state[0] + self.state[1] + self.state[2],
                  self.state[3] + self.state[4] + self.state[5],
                  self.state[6] + self.state[7] + self.state[8],
                  self.state[0] + self.state[4] + self.state[8],
                  self.state[2] + self.state[4] + self.state[6]]
        if 'xxx' in threes:
            return True, 1.0
        elif 'ooo' in threes:
            return True, -1.0
        elif ' ' in self.state:
            return False, 0.0
        else:
            return True, 0.0

    def play(self, move):
        self.state[move] = self.current_player()
        self.players.update_player()
        self.turn += 1

    def reset(self):
        self.state = '         '

    def get_state(self):
        return tuple(self.state)

    def get_moves(self):
        return [pos for pos, char in enumerate(self.state) if char == ' ']

    def pick_move_random(self):
        return np.random.choice(self.get_moves())

    def current_player(self):
        return self.players.current_player()

    def board(self):
        return (self.state[0] + ' | ' + self.state[1] + ' | ' + self.state[2] + '\n' + '---------' + '\n' +
                self.state[3] + ' | ' + self.state[4] + ' | ' + self.state[5] + '\n' + '---------' + '\n' +
                self.state[6] + ' | ' + self.state[7] + ' | ' + self.state[8] + '\n')


class UCTTree(object):
    def __init__(self):
        self.nodes = {}
        self.game = TicTacToe()
        self.nodes[self.game.get_state()]['N'] = 0
        for move in self.game.get_moves():
            self.nodes[(self.game.get_state(), move)]['Q'] = 0
        self.c_puct = .707

    def update_state(self, s, a, r):
        self.nodes[(s, a)]['N'] += 1.0
        self.nodes[(s, a)]['Q'] += r

    def add_state(self, s, a):
        self.nodes[(s, a)] = {'N': 1, 'Q': 0}

    def pick_move_uct(self):
        moves = self.game.get_moves()
        state = self.game.get_state()
        scores = [self.nodes[(state, move)]['Q'] / self.nodes[(state, move)]['N'] +
                  self.c_puct * math.log(self.nodes[(state, move)]['N']) / self.nodes[(state, move)]['N']
                  for move in moves]
        return moves[np.argmax(scores)[0]]

    def calc_score(self, s, a):
        return self.nodes[(s, a)]['Q']

    def tree_rollout(self):
        terminal, r = self.game.check_state()
        if self.game.get_state() in self.nodes.keys():
            if not terminal:
                self.game.play(self.pick_move_uct)
                r = self.tree_rollout()
        else:
            if not terminal:
                self.add_state(self.game.get_state())
                self.game.play(self.game.pick_move_random())
                r = self.default_rollout()
        self.update_state(self.game.get_state(), r)
        return self.game.get_state()['Q']

    def default_rollout(self):
        terminal, r = self.game.check_state()
        if not terminal:
            self.game.play(self.game.pick_move_random())
            r = self.default_rollout()
        self.update_state(self.game.check_state(), r)
        return r

    def search(self):
        pass


if __name__ == "__main__":

    ttt = TicTacToe()
    for game in range(1):
        while not ttt.check_state()[0]:
            print(ttt)
            next_move = np.random.choice(ttt.get_moves())
            ttt.play(next_move)
        print(ttt.board())
        final_state = ttt.check_state()[1]
        if final_state == 1:
            print('VICTORY')
        elif final_state == -1:
            print('DEFEAT')
        elif final_state == 0:
            print('IT\'S A TIE')
        ttt.reset()
