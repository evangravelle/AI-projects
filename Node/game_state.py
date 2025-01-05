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


def are_roads_connected(group1, group2):
    return False


def all_unique(lst):
    return len(lst) == len(set(lst))


"""
Move contains a list of nodes/roads. A node is a pair of whole numbers,
a road is a pair with one number with 0.5.
"""


class GameState:
    def __init__(self, rng_seed=0):
        self.board = create_random_board(rng_seed)
        self.nodes = [[], []]
        self.roads = [[[], []], [[], []]]  # 1st index is player, 2nd index is road group
        self.turn = 1
        self.resources = [
            {"y": 0, "g": 0, "r": 0, "b": 0},
            {"y": 0, "g": 0, "r": 0, "b": 0},
        ]

    def is_move_valid(self, player, move):
        # Inputs must be in the expected format
        for piece in move:
            if not isinstance(piece, tuple):
                print(f"Move invalid because piece {piece} is not a tuple.")
                return False
            if len(piece) != 2:
                print(f"Move invalid because length of piece {piece} is not 2.")
                return False

        if self.turn < 5:
            # First 2 turns for each player should have exactly one node and one road
            if len(move) != 2 or not (piece_is_node(move[0]) ^ piece_is_node(move[1])):
                print(f"Move {move} invalid because first 2 turns should have 1 node and 1 road.")
                return False

            # Roads should have an empty set in first 2 turns for each player
            if len(self.roads[player][0]) != 0 and len(self.roads[player][1]) != 0:
                print(
                    f"Move invalid because both road sets {self.roads[player][0]} and "
                    f"{self.roads[player][1]} are not empty during first 2 turns."
                )
                return False

        # Node and road locations must be unoccupied.
        all_pieces = (
            self.nodes[0] + self.nodes[1] + self.roads[0][0] + self.roads[0][1] + self.roads[1][0] + self.roads[1][1]
        )
        if not all_unique(all_pieces + move):
            print(f"Move invalid because not all pieces {sorted(all_pieces + move)} are unique.")
            return False

        # Nodes must be connected to one of the correct player's road networks.

        # Roads must be connected to one of the correct player's road networks.

        # If we made it this far, move is valid.
        return True

    def move(self, player, move):
        if not self.is_move_valid(player, move):
            raise Exception("Move is not valid.")

        for piece in move:
            if piece_is_node(piece):
                self.nodes[player].append(piece)
            else:
                if self.turn < 5:
                    assert len(self.roads[player][0]) == 0 or len(self.roads[player][1]) == 0
                    if len(self.roads[player][0]) == 0:
                        self.roads[player][0].append(piece)
                    else:
                        if are_roads_connected(self.roads[player][0], piece):
                            self.roads[player][0].append(piece)
                        else:
                            self.roads[player][1].append(piece)
                else:
                    if are_roads_connected(self.roads[player][0], piece):
                        self.roads[player][0].append(piece)
                    else:
                        self.roads[player][1].append(piece)

        self.nodes[player].sort()
        self.roads[player][0].sort()
        self.roads[player][1].sort()
        self.turn += 1

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
