"""
Game state, imagine a grid with vertices where nodes get placed,
(0,2) is the top left node, (0,3) is the top right node.

Board state uses the same grid, the top left vertice is what identifies a tile.

    | |
  | | | |
| | | | | |
  | | | |
    | |
"""

import math
import random

PLAYABLE_ROADS_DICT = {
    # horizontal
    (2.5, 0): {(2, 0.5), (3, 0.5)},
    (1.5, 1): {(1, 1.5), (2, 0.5), (2, 1.5), (2.5, 1)},
    (2.5, 1): {(1.5, 1), (2, 0.5), (2, 1.5), (3, 0.5), (3, 1.5), (3.5, 1)},
    (3.5, 1): {(2.5, 1), (3, 0.5), (3, 1.5), (4, 1.5)},
    (0.5, 2): {(0, 2.5), (1, 1.5), (1, 2.5), (1.5, 2)},
    (1.5, 2): {(0.5, 2), (1, 1.5), (1, 2.5), (2, 1.5), (2, 2.5), (2.5, 2)},
    (2.5, 2): {(1.5, 2), (2, 1.5), (2, 2.5), (3, 1.5), (3, 2.5), (3.5, 2)},
    (3.5, 2): {(2.5, 2), (3, 1.5), (3, 2.5), (4, 1.5), (4, 2.5), (4.5, 2)},
    (4.5, 2): {(3.5, 2), (4, 1.5), (4, 2.5), (5, 2.5)},
    (0.5, 3): {(0, 2.5), (1, 2.5), (1, 3.5), (1.5, 3)},
    (1.5, 3): {(0.5, 3), (1, 2.5), (1, 3.5), (2, 2.5), (2, 3.5), (2.5, 3)},
    (2.5, 3): {(1.5, 3), (2, 2.5), (2, 3.5), (3, 2.5), (3, 3.5), (3.5, 3)},
    (3.5, 3): {(2.5, 3), (3, 2.5), (3, 3.5), (4, 2.5), (4, 3.5), (4.5, 3)},
    (4.5, 3): {(3.5, 3), (4, 2.5), (4, 3.5), (5, 2.5)},
    (1.5, 4): {(1, 3.5), (2, 3.5), (2, 4.5), (2.5, 4)},
    (2.5, 4): {(1.5, 4), (2, 3.5), (2, 4.5), (3, 3.5), (3, 4.5), (3.5, 4)},
    (3.5, 4): {(2.5, 4), (3, 3.5), (3, 4.5), (4, 3.5)},
    (2.5, 5): {(2, 4.5), (3, 4.5)},
    # vertical
    (2, 0.5): {(1.5, 1), (2, 1.5), (2.5, 0), (2.5, 1)},
    (3, 0.5): {(2.5, 0), (2.5, 1), (3, 1.5), (3.5, 1)},
    (1, 1.5): {(0.5, 2), (1, 2.5), (1.5, 1), (1.5, 2)},
    (2, 1.5): {(1.5, 1), (1.5, 2), (2, 0.5), (2, 2.5), (2.5, 1), (2.5, 2)},
    (3, 1.5): {(2.5, 1), (2.5, 2), (3, 0.5), (3, 2.5), (3.5, 1), (3.5, 2)},
    (4, 1.5): {(3.5, 1), (3.5, 2), (4, 2.5), (4.5, 2)},
    (0, 2.5): {(0.5, 2), (0.5, 3)},
    (1, 2.5): {(0.5, 2), (0.5, 3), (1, 1.5), (1, 3.5), (1.5, 2), (1.5, 3)},
    (2, 2.5): {(1.5, 2), (1.5, 3), (2, 1.5), (2, 3.5), (2.5, 2), (2.5, 3)},
    (3, 2.5): {(2.5, 2), (2.5, 3), (3, 1.5), (3, 3.5), (3.5, 2), (3.5, 3)},
    (4, 2.5): {(3.5, 2), (3.5, 3), (4, 1.5), (4, 3.5), (4.5, 2), (4.5, 3)},
    (5, 2.5): {(4.5, 2), (4.5, 3)},
    (1, 3.5): {(0.5, 3), (1, 2.5), (1.5, 3), (1.5, 4)},
    (2, 3.5): {(1.5, 3), (1.5, 4), (2, 2.5), (2, 4.5), (2.5, 3), (2.5, 4)},
    (3, 3.5): {(2.5, 3), (2.5, 4), (3, 2.5), (3, 4.5), (3.5, 3), (3.5, 4)},
    (4, 3.5): {(3.5, 3), (3.5, 4), (4, 2.5), (4.5, 3)},
    (2, 4.5): {(1.5, 4), (2, 3.5), (2.5, 4), (2.5, 5)},
    (3, 4.5): {(2.5, 4), (2.5, 5), (3, 3.5), (3.5, 4)},
}

UNPLAYABLE_NODES = {(0, 0), (1, 0), (4, 0), (5, 0), (0, 1), (5, 1), (0, 4), (5, 4), (0, 5), (1, 5), (4, 5), (5, 5)}

ADJ_TILES_DICT = {
    (2, 0): {(2, 1)},
    (3, 0): {(2, 1)},
    (1, 1): {(1, 2)},
    (2, 1): {(1, 2), (2, 1), (2, 2)},
    (3, 1): {(2, 1), (2, 2), (3, 2)},
    (4, 1): {(3, 2)},
    (0, 2): {(0, 3)},
    (1, 2): {(0, 3), (1, 2), (1, 3)},
    (2, 2): {(1, 2), (1, 3), (2, 2), (2, 3)},
    (3, 2): {(2, 2), (2, 3), (3, 2), (3, 3)},
    (4, 2): {(3, 2), (3, 3), (4, 3)},
    (5, 2): {(4, 3)},
    (0, 3): {(0, 3)},
    (1, 3): {(0, 3), (1, 3), (1, 4)},
    (2, 3): {(1, 3), (1, 4), (2, 3), (2, 4)},
    (3, 3): {(2, 3), (2, 4), (3, 3), (3, 4)},
    (4, 3): {(3, 3), (3, 4), (4, 3)},
    (5, 3): {(4, 3)},
    (1, 4): {(1, 4)},
    (2, 4): {(1, 4), (2, 4), (2, 5)},
    (3, 4): {(2, 4), (2, 5), (3, 4)},
    (4, 4): {(3, 4)},
    (2, 5): {(2, 5)},
    (3, 5): {(2, 5)},
}


def get_playable_roads(roads):
    # There should be 36 keys. Roads are defined left->right or down->up, and sorted lexically x -> y
    playable_roads = set()
    for road in roads:
        playable_roads |= PLAYABLE_ROADS_DICT[road]
    return playable_roads


def get_playable_nodes(roads):
    nodes = set()
    for road in roads:
        nodes.add((math.floor(road[0]), math.floor(road[1])))
        nodes.add((math.ceil(road[0]), math.ceil(road[1])))
    return nodes


def piece_is_node(piece):
    p_sum = piece[0] + piece[1]
    return p_sum - int(p_sum) == 0


def create_random_board(rng_seed):
    tiles = [
        "_",
        "y1",
        "y2",
        "y3",
        "g1",
        "g2",
        "g3",
        "r1",
        "r2",
        "r3",
        "b1",
        "b2",
        "b3",
    ]
    # Upper left location of each tile
    coords = [
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (2, 5),
    ]
    random.seed(rng_seed)
    random.shuffle(tiles)
    board = dict(zip(coords, tiles))
    return board


def are_roads_connected(group1, group2):
    return False


class GameState:
    def __init__(self, rng_seed=0):
        self.board = create_random_board(rng_seed)
        self.player = 0
        self.nodes = [set(), set()]
        self.roads = [[set(), set()], [set(), set()]]  # 1st index is player, 2nd index is road group
        self.turn = 1
        self.resources = [
            {"y": 0, "g": 0, "r": 0, "b": 0},
            {"y": 0, "g": 0, "r": 0, "b": 0},
        ]

    def is_move_valid(self, move):
        # Inputs must be in the expected format
        for piece in move:
            if not isinstance(piece, tuple):
                print(f"Move invalid because piece {piece} is not a tuple.")
                return False
            if len(piece) != 2:
                print(f"Move invalid because length of piece {piece} is not 2.")
                return False

        # First 2 turns for each player should have exactly one node and one road
        if self.turn < 5:
            if len(move) != 2:
                print(f"Move {move} invalid because the first 2 turns should have 2 pieces")
                return False

            move_copy = move.copy()
            move1 = move_copy.pop()
            move2 = move_copy.pop()
            if not (piece_is_node(move1) ^ piece_is_node(move2)):
                print(f"Move {move} invalid because the first 2 turns should have 1 node and 1 road.")
                return False

            # Roads should have an empty set in first 2 turns for each player
            if len(self.roads[self.player][0]) != 0 and len(self.roads[self.player][1]) != 0:
                print(
                    f"Move invalid because both road sets {self.roads[self.player][0]} and "
                    f"{self.roads[self.player][1]} are not empty during first 2 turns."
                )
                return False

        else:
            # After the first 2 turns for each player, nodes and roads cost resources
            cost = {"y": 0, "g": 0, "r": 0, "b": 0}
            for piece in move:
                if piece_is_node(piece):
                    cost["y"] += 2
                    cost["g"] += 2
                else:
                    cost["r"] += 1
                    cost["b"] += 1
            if (
                cost["y"] > self.resources[self.player]["y"]
                or cost["g"] > self.resources[self.player]["g"]
                or cost["r"] > self.resources[self.player]["r"]
                or cost["b"] > self.resources[self.player]["b"]
            ):
                print(
                    f"Move invalid because player {self.player+1} does not have enough resources, available="
                    f"{self.resources[self.player]}, needed={cost}"
                )
                return False

            # Roads and nodes must be connected to one of the correct player's road networks.
            # TODO: If adding more than 1 road, need to verify that one of the roads is connected to the network, then
            #       the 2nd is connected to the new network
            curr_roads = self.roads[self.player][0] | self.roads[self.player][1]
            playable_nodes = get_playable_nodes(curr_roads)
            playable_roads = get_playable_roads(curr_roads)

            for piece in move:
                if piece_is_node(piece):
                    if piece not in playable_nodes:
                        print(f"Move invalid because piece {piece} not in playable nodes {playable_nodes}")
                        return False
                else:
                    if piece not in playable_roads:
                        print(f"Move invalid because piece {piece} not in playable roads {playable_roads}")
                        return False

        # Node and road locations must be unoccupied.
        all_pieces = (
            self.nodes[0] | self.nodes[1] | self.roads[0][0] | self.roads[0][1] | self.roads[1][0] | self.roads[1][1]
        )
        if not all_pieces.isdisjoint(move):
            print(f"Move invalid because not all pieces {all_pieces | move} are unique.")
            return False

        # If we made it this far, move is valid.
        return True

    def spend_resources(self, move):
        print("SPENDING RESOURCES")
        for piece in move:
            if piece_is_node(piece):
                self.resources[self.player]["y"] -= 2
                self.resources[self.player]["g"] -= 2
            else:
                self.resources[self.player]["r"] -= 1
                self.resources[self.player]["b"] -= 1

    def gain_resources(self):
        for player in range(0, 2):
            # First find tiles
            tiles = []
            for node in self.nodes[player]:
                tiles += ADJ_TILES_DICT[node]

            # Now figure out what resources to get
            # TODO: CHECK THE STATE OF TILES (overloaded, nuked, or surrounded)
            resources_str = "".join([self.board[tile][0] for tile in tiles])
            print(resources_str)
            for ch in "ygrb":
                self.resources[player][ch] = resources_str.count(ch)

    def update_game_state(self, move):
        # Update roads and nodes
        for piece in move:
            if piece_is_node(piece):
                self.nodes[self.player].add(piece)
            else:
                if self.turn < 5:
                    if len(self.roads[self.player][0]) == 0:
                        self.roads[self.player][0].add(piece)
                    else:
                        if are_roads_connected(self.roads[self.player][0], piece):
                            self.roads[self.player][0].add(piece)
                        else:
                            self.roads[self.player][1].add(piece)
                else:
                    if are_roads_connected(self.roads[self.player][0], piece):
                        self.roads[self.player][0].add(piece)
                    else:
                        self.roads[self.player][1].add(piece)

        # Check if both groups are now connected
        if self.roads[self.player][1] and are_roads_connected(self.roads[self.player][0], self.roads[self.player][1]):
            self.roads[self.player][0] |= self.roads[self.player][1]
            self.roads[self.player][1] = set()

        self.player = (self.player + 1) % 2

    def move(self, move):
        """
        Move contains a list of nodes/roads. A node is a pair of whole numbers,
        a road is a pair with one number with 0.5.
        """

        # Check if move is valid given game state.
        if not self.is_move_valid(move):
            raise Exception("Move is not valid.")

        # Spend resources for the move.
        if self.turn >= 5:
            self.spend_resources(move)

        # Update game state.
        self.update_game_state(move)

        # If player 2 just moved, everyone gains resources.
        if self.turn >= 4 and self.turn % 2 == 0:
            self.gain_resources()

        # Clean up game state, finish turn.
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

    def get_available_moves(self):
        pass
