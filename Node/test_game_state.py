import FreeSimpleGUI as sg
import game_state
import gui
import unittest


GUI_ENABLED = False


def draw_road_adj_roads_event_loop(window):
    gui.draw_board(window)
    graph = window["-GRAPH-"]

    road_ids = []
    road_keys = game_state.PLAYABLE_ROADS_DICT.keys()
    road_keys_it = iter(road_keys)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "-BUTTON-":
            try:
                road_key = next(road_keys_it)
                adj_roads = game_state.PLAYABLE_ROADS_DICT[road_key]
                clear_pieces(graph, road_ids)
                road_ids = []
                road_ids = draw_roads(graph, {road_key}, road_ids, "white")
                road_ids = draw_roads(graph, adj_roads, road_ids, "orange")
            except StopIteration:
                break

    window.close()


def draw_tile_adj_nodes_event_loop(window):
    gui.draw_board(window)
    graph = window["-GRAPH-"]

    piece_ids = []
    tile_keys = game_state.TILE_TO_ADJ_NODES_DICT.keys()
    tile_keys_it = iter(tile_keys)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "-BUTTON-":
            try:
                tile_key = next(tile_keys_it)
                adj_nodes = game_state.TILE_TO_ADJ_NODES_DICT[tile_key]
                clear_pieces(graph, piece_ids)
                piece_ids = []
                piece_ids = draw_tiles(graph, {tile_key}, piece_ids, "white")
                piece_ids = draw_nodes(graph, adj_nodes, piece_ids, "orange")
            except StopIteration:
                break

    window.close()


def draw_node_adj_tiles_event_loop(window):
    gui.draw_board(window)
    graph = window["-GRAPH-"]

    piece_ids = []
    node_keys = game_state.NODE_TO_ADJ_TILES_DICT.keys()
    node_keys_it = iter(node_keys)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "-BUTTON-":
            try:
                node_key = next(node_keys_it)
                adj_tiles = game_state.NODE_TO_ADJ_TILES_DICT[node_key]
                clear_pieces(graph, piece_ids)
                piece_ids = []
                piece_ids = draw_nodes(graph, {node_key}, piece_ids, "white")
                piece_ids = draw_tiles(graph, adj_tiles, piece_ids, "orange")
            except StopIteration:
                break

    window.close()


def draw_road_adj_tiles_event_loop(window):
    gui.draw_board(window)
    graph = window["-GRAPH-"]
    piece_ids = []
    road_keys = game_state.ROAD_TO_ADJ_TILES_DICT.keys()
    road_keys_it = iter(road_keys)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "-BUTTON-":
            try:
                road_key = next(road_keys_it)
                adj_tiles = game_state.ROAD_TO_ADJ_TILES_DICT[road_key]
                clear_pieces(graph, piece_ids)
                piece_ids = []
                piece_ids = draw_roads(graph, {road_key}, piece_ids, "white")
                piece_ids = draw_tiles(graph, adj_tiles, piece_ids, "orange")
            except StopIteration:
                break

    window.close()


def clear_pieces(graph, piece_ids):
    for piece_id in piece_ids:
        graph.delete_figure(piece_id)


def draw_roads(graph, roads, road_ids, color):
    for road in roads:
        road_id = gui.draw_road(graph, road, color)
        road_ids.append(road_id)
    return road_ids


def draw_nodes(graph, nodes, node_ids, color):
    for node in nodes:
        node_id = gui.draw_node(graph, node, color)
        node_ids.append(node_id)
    return node_ids


def draw_tiles(graph, tiles, tile_ids, color):
    for tile in tiles:
        tile_id = gui.draw_node(graph, (tile[0] + 0.5, tile[1] - 0.5), color)
        tile_ids.append(tile_id)
    return tile_ids


class TestGameState(unittest.TestCase):

    def test_invalid_initial_moves(self):
        state = game_state.GameState()
        prev_player = state.player
        state.move({(0, 2), (0, 2.5)})
        self.assertEqual(state.nodes[prev_player], {(0, 2)})
        self.assertEqual(state.roads[prev_player], [{(0, 2.5)}, set()])
        # Invalid inputs
        with self.assertRaises(Exception):
            state.move({(), (0, 2.5)})
        with self.assertRaises(Exception):
            state.move({(1,), ()})
        with self.assertRaises(Exception):
            state.move({(1, 1), ()})
        with self.assertRaises(Exception):
            state.move({(1, 1), (3,)})
        with self.assertRaises(Exception):
            state.move({(1, 1), (1, 2, 3)})
        # Moves that are already on the board
        with self.assertRaises(Exception):
            state.move({(0, 2), (1, 2.5)})
        with self.assertRaises(Exception):
            state.move({(0, 2), (1, 2.5)})
        with self.assertRaises(Exception):
            state.move({(1, 2), (0, 2.5)})
        with self.assertRaises(Exception):
            state.move({(1, 2), (0, 2.5)})

    def test_valid_initial_moves(self):
        state = game_state.GameState()
        prev_player = state.player
        state.move({(2, 2), (2, 1.5)})
        self.assertEqual(state.nodes[prev_player], {(2, 2)})
        self.assertEqual(state.roads[prev_player], [{(2, 1.5)}, set()])
        prev_player = state.player
        state.move({(3, 2), (3.5, 2)})
        self.assertEqual(state.nodes[prev_player], {(3, 2)})
        self.assertEqual(state.roads[prev_player], [{(3.5, 2)}, set()])
        prev_player = state.player
        state.move({(3, 3), (3, 3.5)})
        self.assertEqual(state.nodes[prev_player], {(2, 2), (3, 3)})
        self.assertEqual(state.roads[prev_player], [{(2, 1.5)}, {(3, 3.5)}])
        prev_player = state.player
        state.move({(2, 3), (1.5, 3)})
        self.assertEqual(state.nodes[prev_player], {(2, 3), (3, 2)})
        self.assertEqual(state.roads[prev_player], [{(3.5, 2)}, {(1.5, 3)}])

    def test_invalid_moves(self):
        state = game_state.GameState()
        state.move({(2, 2), (2, 1.5)})
        state.move({(3, 2), (3.5, 2)})
        state.move({(3, 3), (3, 3.5)})
        state.move({(2, 3), (1.5, 3)})
        state.resources[0] = {"y": 100, "g": 100, "r": 100, "b": 100}
        state.resources[1] = {"y": 100, "g": 100, "r": 100, "b": 100}
        # Invalid inputs
        with self.assertRaises(Exception):
            state.move({(), (0, 2.5)})
        with self.assertRaises(Exception):
            state.move({(1,), ()})
        with self.assertRaises(Exception):
            state.move({(1, 1), ()})
        with self.assertRaises(Exception):
            state.move({(1, 1), (3,)})
        with self.assertRaises(Exception):
            state.move({(1, 1), (1, 2, 3)})
        # Moves that are already on the board
        with self.assertRaises(Exception):
            state.move({(2, 2), (2.5, 2)})
        with self.assertRaises(Exception):
            state.move({(3, 2), (3, 2.5)})
        with self.assertRaises(Exception):
            state.move({(2, 1), (2, 1.5)})
        with self.assertRaises(Exception):
            state.move({(3, 4), (3, 3.5)})
        # Moves that are not connected to the correct network
        with self.assertRaises(Exception):
            state.move({(4, 4), (4, 3.5)})
        with self.assertRaises(Exception):
            state.move({(4, 4), (4, 3.5)})
        with self.assertRaises(Exception):
            state.move({(3, 1), (2.5, 1)})
        with self.assertRaises(Exception):
            state.move({(3, 1), (3, 1.5)})

    def test_score(self):
        state = game_state.GameState()
        score = state.score
        self.assertEqual(score, [0, 0])

    def test_draw_road_adjacent_roads(self):
        if GUI_ENABLED:
            window = gui.create_window("Draw Road's Adjacent Roads")
            draw_road_adj_roads_event_loop(window)

    def test_draw_tile_adjacent_nodes(self):
        if GUI_ENABLED:
            window = gui.create_window("Draw Tile's Adjacent Nodes")
            draw_tile_adj_nodes_event_loop(window)

    def test_draw_node_adjacent_tiles(self):
        if GUI_ENABLED:
            window = gui.create_window("Draw Node's Adjacent Tiles")
            draw_node_adj_tiles_event_loop(window)

    def test_draw_road_adjacent_tiles(self):
        if GUI_ENABLED:
            window = gui.create_window("Draw Road's Adjacent Tiles")
            draw_road_adj_tiles_event_loop(window)

    def test_get_playable_nodes(self):
        roads = set()
        nodes = game_state.get_playable_nodes(roads)
        expected_nodes = set()
        self.assertEqual(nodes, expected_nodes)

        roads = {(2.5, 2), (3, 2.5)}
        nodes = game_state.get_playable_nodes(roads)
        expected_nodes = {(2, 2), (3, 2), (3, 3)}
        self.assertEqual(nodes, expected_nodes)

        roads = {(0, 2.5), (0.5, 2), (0.5, 3), (1, 2.5)}
        nodes = game_state.get_playable_nodes(roads)
        expected_nodes = {(0, 2), (0, 3), (1, 2), (1, 3)}
        self.assertEqual(nodes, expected_nodes)

    def test_get_available_moves(self):
        pass


if __name__ == "__main__":
    unittest.main()
