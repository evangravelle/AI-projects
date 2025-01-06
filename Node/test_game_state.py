import FreeSimpleGUI as sg
import game_state
import gui
import unittest


GUI_ENABLED = False


def draw_roads_event_loop(window, button_str):
    graph = None
    mode = "INIT"
    road_ids = []
    road_keys = game_state.PLAYABLE_ROADS_DICT.keys()
    road_keys_it = iter(road_keys)
    while True:
        # sg.window.Window.read()
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if mode == "INIT":
            gui.draw_board(window)
            graph = window["-GRAPH-"]
            button = window["-BUTTON-"]
            button.update(text=button_str)
            mode = "DRAW"
        elif mode == "DRAW" and event == "-BUTTON-":
            try:
                road_key = next(road_keys_it)
                adj_roads = game_state.PLAYABLE_ROADS_DICT[road_key]
                clear_roads(graph, road_ids)
                road_ids = [gui.draw_road(graph, road_key, "white")]
                road_ids = draw_roads(graph, adj_roads, road_ids)
            except StopIteration:
                break

    window.close()


def draw_neighboring_tiles_event_loop(window, button_str):
    graph = None
    mode = "INIT"
    tile_ids = []
    tile_keys = game_state.ADJ_TILES_DICT.keys()
    tile_keys_it = iter(tile_keys)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if mode == "INIT":
            gui.draw_board(window)
            graph = window["-GRAPH-"]
            button = window["-BUTTON-"]
            button.update(text=button_str)
            mode = "DRAW"
        elif mode == "DRAW" and event == "-BUTTON-":
            try:
                tile_key = next(tile_keys_it)
                adj_tiles = game_state.ADJ_TILES_DICT[tile_key]
                clear_roads(graph, tile_ids)
                tile_ids = [gui.draw_node(graph, tile_key, "white")]
                tile_ids = draw_nodes(graph, adj_tiles, tile_ids)
            except StopIteration:
                break


def clear_roads(graph, road_ids):
    for road_id in road_ids:
        graph.delete_figure(road_id)


def draw_roads(graph, roads, road_ids):
    for road in roads:
        road_id = gui.draw_road(graph, road, "orange")
        road_ids.append(road_id)
    return road_ids


def draw_nodes(graph, nodes, node_ids):
    for node in nodes:
        node_id = gui.draw_node(graph, (node[0] + 0.5, node[1] - 0.5), "orange")
        node_ids.append(node_id)
    return node_ids


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
        score = state.get_score()
        self.assertEqual(score, (0, 0))

    def test_get_playable_roads(self):
        if GUI_ENABLED:
            button_str = "Draw Adjacent Roads"
            window = gui.create_window()
            draw_roads_event_loop(window, button_str)

    def test_get_neighboring_tiles(self):
        if GUI_ENABLED:
            button_str = "Draw Neighboring Tiles"
            window = gui.create_window()
            draw_neighboring_tiles_event_loop(window, button_str)

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
