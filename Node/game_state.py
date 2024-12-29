"""
Game state, imagine a grid with vertices where nodes get placed,
(0,2) is the top left node, (0,3) is the top right node.

Board state uses the same grid, the top left vertice is what identifies a square.

    | |
  | | | |
| | | | | |
  | | | |
    | |
"""

import random

def create_random_board(rng_seed):
    tiles = ['blank', 'Y1', 'Y2', 'Y3', 'G1', 'G2', 'G3', 'R1', 'R2', 'R3', 'B1', 'B2', 'B3']
    spots = [(0,2), (1,1), (1,2), (1,3), (2,0), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (4,2)]
    random.seed(rng_seed)
    random.shuffle(tiles)
    board = dict(zip(spots, tiles))
    return board

class GameState:
    def __init__(self, rng_seed = 0):
        self.board = create_random_board(rng_seed)
        self.nodes = []
        self.roads = []

    """
    Move contains a list of nodes/roads. A node is a pair, a road is a pair of pairs
    """
    def update(self, move):
        for piece in move:
            assert (len(move) == 1 or len(move) == 2)
            if len(move) == 1:
                self.nodes.append(piece)
            elif len(move) == 2:
                self.roads.append(piece)
            else:
                raise RuntimeError("A piece in a move should be length 1 or 2.")
