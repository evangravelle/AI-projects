""" Tic Tac Toe
Written by Evan Gravelle
01/22/2018 """

import math
import numpy as np
np.set_printoptions(precision=2)


class Player(object):
    def __init__(self):
        """Initializes player 1 to x."""
        self.player = 1
        self.players = 'xo'

    def current_player(self):
        """Returns current player."""
        return self.players[self.player - 1]

    def update_player(self):
        """Updates the player, returns new player."""
        self.player = self.player % 2 + 1
        return self.current_player()


class TicTacToe(object):
    def __init__(self):
        """Initializes an empty board and a player."""
        self.state = list('         ')
        self.players = Player()
        self.turn = 0

    def __str__(self):
        """Prints the board in a human readable format."""
        return (self.board() + '\nTurn ' + str(self.turn + 1) + ', ' +
                self.players.current_player().upper() + "\'s turn\n")

    def check_state(self):
        """Returns tuple of (is_terminal, reward).
        TODO: if str cat is slow, can use ints 0,1,4 and addition to represent state."""
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
        """Plays a move."""
        self.state[move] = self.current_player()
        self.players.update_player()
        self.turn += 1

    def reset(self, s=None):
        """Resets the board."""
        if s:
            self.state = s.copy()
            self.turn = 9 - s.count(' ')
        else:
            self.state = list('         ')
            self.turn = 0

    def get_state(self):
        """Returns a tuple of the board state, to be used as a hash key."""
        return tuple(self.state)

    def get_moves(self, s=None):
        """Returns the set of legal moves, expressed as a list of indices."""
        if s is None:
            return [pos for pos, char in enumerate(self.state) if char == ' ']
        else:
            return [pos for pos, char in enumerate(s) if char == ' ']

    def pick_move_random(self):
        """Returns a random legal move."""
        return np.random.choice(self.get_moves())

    def current_player(self):
        """Returns the current player."""
        return self.players.current_player()

    def board(self):
        """Returns the board state in a human readable format."""
        return (self.state[0] + ' | ' + self.state[1] + ' | ' + self.state[2] + '\n' + '---------' + '\n' +
                self.state[3] + ' | ' + self.state[4] + ' | ' + self.state[5] + '\n' + '---------' + '\n' +
                self.state[6] + ' | ' + self.state[7] + ' | ' + self.state[8] + '\n')


class UCTTree(object):
    def __init__(self):
        """Initializes a tic-tac-toe game and the root of the tree."""
        self.game = TicTacToe()
        self.nodes = {}
        self.add_node(self.game.get_state())
        self.cp_uct = .707

    def add_node(self, s):
        """Adds a node to the tree."""
        self.nodes[s] = {'N': 0}
        for a in self.game.get_moves(s):
            self.nodes[s][a] = {'Q': 0.0, 'N': 0}

    def update_state(self, s, a, r):
        """Updates the count and reward of a state-action pair."""
        self.nodes[s]['N'] += 1
        self.nodes[s][a]['N'] += 1
        self.nodes[s][a]['Q'] += r

    def pick_move_uct(self):
        """Chooses a move using the UCT algorithm. If some moves haven't been expanded yet,
        this chooses randomly among them."""
        moves = self.game.get_moves()
        s = self.game.get_state()
        visits = np.array([self.nodes[s][a]['N'] for a in moves])
        if not np.all(visits):
            return moves[np.random.choice(np.argwhere(visits == 0).flatten())]
        else:
            scores = [self.nodes[s][a]['Q'] / self.nodes[s][a]['N'] +
                      self.cp_uct * math.log(self.nodes[s][a]['N']) / self.nodes[s]['N']
                      for a in moves]
            return moves[np.random.choice(np.argwhere(scores == np.max(scores)).flatten())]

    def tree_rollout(self, alg='uct'):
        """Executes rollout using UCT algorithm, recursively updating rewards and counts."""
        if alg == 'uct':
            move_dict = {alg: self.pick_move_uct}
        else:
            move_dict = {alg: self.game.pick_move_random}
        s = self.game.get_state()
        p = self.game.current_player()
        terminal, r = self.game.check_state()
        if s in self.nodes.keys():
            if not terminal:
                a = move_dict[alg]()
                self.game.play(a)
                r = self.tree_rollout()
                self.update_state(s, a, r)
            else:
                if p == 'x':
                    return r
                else:
                    return -r
        else:
            if not terminal:
                self.add_node(s)
                a = move_dict[alg]()
                self.game.play(a)
                r = self.tree_rollout()
                self.update_state(s, a, r)
            else:
                if p == 'x':
                    return r
                else:
                    return -r
        return r

    def best_move(self):
        """Returns the move which has the highest visit count, a proxy for best move."""
        counts = np.array([(a, self.nodes[self.game.get_state()][a]['N'])
                          for a in self.nodes[self.game.get_state()] if isinstance(a, int)])
        return np.random.choice(counts[np.argwhere(counts[:, 1] == np.max(counts[:, 1])), 0].flatten())


def play_random_game():
    """Plays random game of tic-tac-toe."""
    ttt = TicTacToe()
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


def play_uct_game():
    """Plays game of tic-tac-toe using UCT."""
    ttt = TicTacToe()
    uct = UCTTree()
    while not ttt.check_state()[0]:
        print(ttt)
        for i in range(1000):
            uct.tree_rollout()
            uct.game.reset(ttt.state)
        next_move = uct.best_move()
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


def play_uct_with_human():
    """Plays game of tic-tac-toe vs human using UCT."""
    ttt = TicTacToe()
    uct = UCTTree()
    while not ttt.check_state()[0]:
        print(ttt)
        ttt.play(input('Enter a move:'))
        print(ttt)
        for i in range(1000):
            uct.tree_rollout()
            uct.game.reset(ttt.state)
        next_move = uct.best_move()
        ttt.play(next_move)
    print(ttt)
    final_state = ttt.check_state()[1]
    if final_state == 1:
        print('VICTORY')
    elif final_state == -1:
        print('DEFEAT')
    elif final_state == 0:
        print('IT\'S A TIE')
    ttt.reset()


if __name__ == "__main__":
    play_uct_game()
