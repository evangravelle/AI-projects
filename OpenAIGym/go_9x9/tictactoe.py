""" Tic Tac Toe
Written by Evan Gravelle
01/22/2018 """

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
        self.state = list('.........')
        self.player = Player()
        self.turn = 0

    def __str__(self):
        return ('Turn ' + str(self.turn).upper() + ', ' + self.player.players[self.player.player] + ' gets to go\n\n' +
                self.board())

    def check_state(self):
        threes = [(self.state[0], self.state[3], self.state[6]),
                  (self.state[1], self.state[4], self.state[7]),
                  (self.state[2], self.state[5], self.state[8]),
                  (self.state[0], self.state[1], self.state[2]),
                  (self.state[3], self.state[4], self.state[5]),
                  (self.state[6], self.state[7], self.state[8]),
                  (self.state[0], self.state[4], self.state[8]),
                  (self.state[2], self.state[4], self.state[6])]
        if 'xxx' in threes:
            return 'x'
        elif 'ooo' in threes:
            return 'o'
        elif '.' in self.state:
            return 'p'
        else:
            return 'c'

    def play(self, move):
        self.state[move] = self.player.players[self.player.player]
        self.player.update_player()
        self.turn += 1

    def reset(self):
        self.state = '.........'

    def get_hash(self):
        return hash(tuple(self.state))

    def get_moves(self):
        return [pos for pos, char in enumerate(self.state) if char == '.']

    def current_player(self):
        return self.player.current_player()

    def board(self):
        return (self.state[0] + ' | ' + self.state[1] + ' | ' + self.state[2] + '\n' + '---------' + '\n' +
                self.state[3] + ' | ' + self.state[4] + ' | ' + self.state[5] + '\n' + '---------' + '\n' +
                self.state[6] + ' | ' + self.state[7] + ' | ' + self.state[8] + '\n')



class Tree(object):
    def __init__(self):
        self.nodes = {}

    def add_state(self, s):
        if s in self.nodes.keys():
            self.nodes[s]['count'] += 1
            self.nodes[s]['value'] += 1
            self.nodes[s]['children'] += 'timmy'
        else:
            node = {'count': 1, 'value': 0, 'children': []}
            self.nodes[s] = node


if __name__ == "__main__":

    ttt = TicTacToe()
    for game in range(1):
        while ttt.check_state() == 'p':
            print(ttt)
            move = np.random.choice(ttt.get_moves())
            ttt.play(move)
        print(ttt.board())
        final_state = ttt.check_state()
        if final_state == 'x':
            print('VICTORY')
        elif final_state == 'o':
            print('DEFEAT')
        elif final_state == 'c':
            print('IT\'S A TIE')
        ttt.reset()
