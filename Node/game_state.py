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


class GameState:
    def __init__(self, rng_seed=0):
        self.board = create_random_board(rng_seed)
        self.nodes = {"p1": [], "p2": []}
        self.roads = {"p1": [], "p2": []}
        self.p1 = {"y": 0, "g": 0, "r": 0, "b": 0}
        self.p2 = {"y": 0, "g": 0, "r": 0, "b": 0}

    """
    Move contains a list of nodes/roads. A node is a pair, a road is a pair of pairs
    """

    def update(self, player, move):
        for piece in move:
            assert isinstance(piece, tuple) and len(piece) == 2
            if not isinstance(piece[0], tuple):
                assert len(piece) == 2
                self.nodes[player].append(piece)
            else:
                assert len(piece[0]) == 2 and len(piece[1]) == 2
                self.roads[player].append(piece)

    def get_score(self):
        p1_score = len(self.nodes["p1"])
        p2_score = len(self.nodes["p2"])

        # Check the number of groups before computing the max group size
        if len(self.roads["p1"]) == 2:
            p1_max_group_size = max(len(self.roads["p1"][0]), len(self.roads["p1"][1]))
        elif len(self.roads["p1"]) > 0:
            p1_max_group_size = len(self.roads["p1"][0])
        else:
            p1_max_group_size = 0

        if len(self.roads["p2"]) == 2:
            p2_max_group_size = max(len(self.roads["p2"][0]), len(self.roads["p2"][1]))
        elif len(self.roads["p1"]) > 0:
            p2_max_group_size = len(self.roads["p2"][0])
        else:
            p2_max_group_size = 0

        if p1_max_group_size > p2_max_group_size:
            p1_score += 2
        elif p2_max_group_size > p1_max_group_size:
            p2_score += 2
        return p1_score, p2_score

    @staticmethod
    def get_available_moves(game_state):
        pass
