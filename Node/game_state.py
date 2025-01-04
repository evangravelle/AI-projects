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

import math
import random

# There should be 36 keys. Roads are defined left->right or down->up, and sorted lexically x -> y
playable_roads = {
    # horizontal
    (2.5, 0): [(2, 0.5), (3, 0.5)],
    (1.5, 1): [(1, 1.5), (2, 0.5), (2, 1.5), (2.5, 1)],
    (2.5, 1): [(1.5, 1), (2, 0.5), (2, 1.5), (3, 0.5), (3, 1.5), (3.5, 1)],
    (3.5, 1): [(2.5, 1), (3, 0.5), (3, 1.5), (4, 1.5)],
    (0.5, 2): [(0, 2.5), (1, 1.5), (1, 2.5), (1.5, 2)],
    (1.5, 2): [(0.5, 2), (1, 1.5), (1, 2.5), (2, 1.5), (2, 2.5), (2.5, 2)],
    (2.5, 2): [(1.5, 2), (2, 1.5), (2, 2.5), (3, 1.5), (3, 2.5), (3.5, 2)],
    (3.5, 2): [(2.5, 2), (3, 1.5), (3, 2.5), (4, 1.5), (4, 2.5), (4.5, 2)],
    (4.5, 2): [(3.5, 2), (4, 1.5), (4, 2.5), (5, 2.5)],
    (0.5, 3): [(0, 2.5), (1, 2.5), (1, 3.5), (1.5, 3)],
    (1.5, 3): [(0.5, 3), (1, 2.5), (1, 3.5), (2, 2.5), (2, 3.5), (2.5, 3)],
    (2.5, 3): [(1.5, 3), (2, 2.5), (2, 3.5), (3, 2.5), (3, 3.5), (3.5, 3)],
    (3.5, 3): [(2.5, 3), (3, 2.5), (3, 3.5), (4, 2.5), (4, 3.5), (4.5, 3)],
    (4.5, 3): [(3.5, 3), (4, 2.5), (4, 3.5), (5, 2.5)],
    (1.5, 4): [(1, 3.5), (2, 3.5), (2, 4.5), (2.5, 4)],
    (2.5, 4): [(1.5, 4), (2, 3.5), (2, 4.5), (3, 3.5), (3, 4.5), (3.5, 4)],
    (3.5, 4): [(2.5, 4), (3, 3.5), (3, 4.5), (4, 3.5)],
    (2.5, 5): [(2, 4.5), (3, 4.5)],
    # vertical
    (2, 0.5): [(1.5, 1), (2, 1.5), (2.5, 0), (2.5, 1)],
    (3, 0.5): [(2.5, 0), (2.5, 1), (3, 1.5), (3.5, 1)],
    (1, 1.5): [(0.5, 2), (1, 2.5), (1.5, 1), (1.5, 2)],
    (2, 1.5): [(1.5, 1), (1.5, 2), (2, 0.5), (2, 2.5), (2.5, 1), (2.5, 2)],
    (3, 1.5): [(2.5, 1), (2.5, 2), (3, 0.5), (3, 2.5), (3.5, 1), (3.5, 2)],
    (4, 1.5): [(3.5, 1), (3.5, 2), (4, 2.5), (4.5, 2)],
    (0, 2.5): [(0.5, 2), (0.5, 3)],
    (1, 2.5): [(0.5, 2), (0.5, 3), (1, 1.5), (1, 3.5), (1.5, 2), (1.5, 3)],
    (2, 2.5): [(1.5, 2), (1.5, 3), (2, 1.5), (2, 3.5), (2.5, 2), (2.5, 3)],
    (3, 2.5): [(2.5, 2), (2.5, 3), (3, 1.5), (3, 3.5), (3.5, 2), (3.5, 3)],
    (4, 2.5): [(3.5, 2), (3.5, 3), (4, 1.5), (4, 3.5), (4.5, 2), (4.5, 3)],
    (5, 2.5): [(4.5, 2), (4.5, 3)],
    (1, 3.5): [(0.5, 3), (1, 2.5), (1.5, 3), (1.5, 4)],
    (2, 3.5): [(1.5, 3), (1.5, 4), (2, 2.5), (2, 4.5), (2.5, 3), (2.5, 4)],
    (3, 3.5): [(2.5, 3), (2.5, 4), (3, 2.5), (3, 4.5), (3.5, 3), (3.5, 4)],
    (4, 3.5): [(3.5, 3), (3.5, 4), (4, 2.5), (4.5, 3)],
    (2, 4.5): [(1.5, 4), (2, 3.5), (2.5, 4), (2.5, 5)],
    (3, 4.5): [(2.5, 4), (2.5, 5), (3, 3.5), (3.5, 4)],
}


def get_playable_nodes(roads):
    nodes = []
    for road in roads:
        nodes.append((math.floor(road[0]), math.floor(road[1])))
        nodes.append((math.ceil(road[0]), math.ceil(road[1])))
    unique_list = list(set(nodes))
    # lexical tuple sorting
    sorted_list = sorted(unique_list, key=lambda tup: (tup[0], tup[1]))
    return sorted_list


def piece_is_node(piece):
    p_sum = piece[0] + piece[1]
    return p_sum - int(p_sum) == 0


def create_random_board(rng_seed):
    tiles = [
        "blank",
        "Y1",
        "Y2",
        "Y3",
        "G1",
        "G2",
        "G3",
        "R1",
        "R2",
        "R3",
        "B1",
        "B2",
        "B3",
    ]
    spots = [
        (2, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (2, 4),
    ]
    random.seed(rng_seed)
    random.shuffle(tiles)
    board = dict(zip(spots, tiles))
    return board


"""
Move contains a list of nodes/roads. A node is a pair of whole numbers,
a road is a pair with one number with 0.5.
"""
class GameState:
    def __init__(self, rng_seed=0):
        self.board = create_random_board(rng_seed)
        self.nodes = [[], []]
        self.roads = [[[]], [[]]]  # First index is player, 2nd index is road group
        self.resources = [
            {"y": 0, "g": 0, "r": 0, "b": 0},
            {"y": 0, "g": 0, "r": 0, "b": 0},
        ]

    def initial_move(self, player, move, number):
        for piece in move:
            assert isinstance(piece, tuple) and len(piece) == 2
            if piece_is_node(piece):
                self.nodes[player].append(piece)
            else:
                self.roads[player][number].append(piece)

    def update(self, player, move):
        for piece in move:
            assert isinstance(piece, tuple) and len(piece) == 2
            if piece_is_node(piece):
                self.nodes[player].append(piece)
            else:
                # TODO: figure out which group to add this road to
                self.roads[player].append(piece)

    def get_score(self):
        scores = (len(self.nodes[0]), len(self.nodes[1]))

        # Check the number of groups before computing the max group size
        max_group_sizes = [0, 0]
        for p in range(0, 2):
            if len(self.roads[p]) == 2:
                max_group_sizes[p] = max(len(self.roads[p][0]), len(self.roads[p][1]))
            elif len(self.roads[p]) == 1:
                max_group_sizes[p] = len(self.roads[p][0])
            else:
                max_group_sizes[p] = 0

        if max_group_sizes[0] > max_group_sizes[1]:
            scores[0] += 2
        elif max_group_sizes[0] > max_group_sizes[1]:
            scores[1] += 2
        return scores

    @staticmethod
    def get_available_moves(game_state):
        pass
