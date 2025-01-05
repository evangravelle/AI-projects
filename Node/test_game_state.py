import FreeSimpleGUI as sg
import game_state
import gui
import unittest


GUI_ENABLED = False


def draw_roads_event_loop(window, button_str):
    graph = None
    mode = "INIT"
    road_ids = []
    road_keys = game_state.playable_roads.keys()
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
            road_key = next(road_keys_it)
            adj_roads = game_state.playable_roads[road_key]
            clear_roads(graph, road_ids)
            road_ids = [gui.draw_road(graph, road_key, "white")]
            road_ids = draw_roads(graph, adj_roads, road_ids)

    window.close()


def clear_roads(graph, road_ids):
    for road_id in road_ids:
        graph.delete_figure(road_id)


def draw_roads(graph, roads, road_ids):
    for road in roads:
        road_id = gui.draw_road(graph, road, "orange")
        road_ids.append(road_id)
    return road_ids


class TestGameState(unittest.TestCase):

    def test_invalid_initial_moves(self):
        state = game_state.GameState()
        player = 0
        state.move(player=player, move=[(0, 2), (0, 2.5)])
        self.assertEqual(state.nodes[player], [(0, 2)])
        self.assertEqual(state.roads[player], [[(0, 2.5)], []])
        with self.assertRaises(Exception):
            # Invalid inputs
            state.move(player=1, move=[(), (0, 2.5)])
            state.move(player=2, move=[(1,), ()])
            state.move(player=1, move=[(1, 1), ()])
            state.move(player=2, move=[(1, 1), (3,)])
            state.move(player=1, move=[(1, 1), (1, 2, 3)])
            # Moves that are already on the board
            state.move(player=1, move=[(0, 2), (1, 2.5)])
            state.move(player=2, move=[(0, 2), (1, 2.5)])
            state.move(player=1, move=[(1, 2), (0, 2.5)])
            state.move(player=2, move=[(1, 2), (0, 2.5)])

    def test_valid_initial_moves(self):
        state = game_state.GameState()
        player = 0
        state.move(player=player, move=[(2, 2), (2.5, 2)])
        self.assertEqual(state.nodes[player], [(2, 2)])
        self.assertEqual(state.roads[player], [[(2.5, 2)], []])
        player = 1
        state.move(player=player, move=[(3, 2), (3, 2.5)])
        self.assertEqual(state.nodes[player], [(3, 2)])
        self.assertEqual(state.roads[player], [[(3, 2.5)], []])
        player = 0
        state.move(player=player, move=[(3, 3), (2.5, 3)])
        self.assertEqual(state.nodes[player], [(2, 2), (3, 3)])
        self.assertEqual(state.roads[player], [[(2.5, 2)], [(2.5, 3)]])
        player = 1
        state.move(player=player, move=[(2, 3), (2, 2.5)])
        self.assertEqual(state.nodes[player], [(2, 3), (3, 2)])
        self.assertEqual(state.roads[player], [[(3, 2.5)], [(2, 2.5)]])

    def test_invalid_moves(self):
        state = game_state.GameState()
        player = 0
        state.move(player=player, move=[(2, 2), (2.5, 2)])
        player = 1
        state.move(player=player, move=[(3, 2), (3, 2.5)])
        player = 0
        state.move(player=player, move=[(3, 3), (2.5, 3)])
        player = 1
        state.move(player=player, move=[(2, 3), (2, 2.5)])
        with self.assertRaises(Exception):
            # Invalid inputs
            state.move(player=1, move=[(), (0, 2.5)])
            state.move(player=2, move=[(1,), ()])
            state.move(player=1, move=[(1, 1), ()])
            state.move(player=2, move=[(1, 1), (3,)])
            state.move(player=1, move=[(1, 1), (1, 2, 3)])
            # Moves that are already on the board
            state.move(player=1, move=[(0, 2), (1, 2.5)])
            state.move(player=2, move=[(0, 2), (1, 2.5)])
            state.move(player=1, move=[(1, 2), (0, 2.5)])
            state.move(player=2, move=[(1, 2), (0, 2.5)])
            # Moves that are not connected to the network
            state.move(player=1, move=[(4, 4), (3.5, 4)])
            state.move(player=2, move=[(4, 4), (4, 3.5)])
            state.move(player=1, move=[(3, 1), (2.5, 1)])
            state.move(player=2, move=[(3, 1), (3.5, 1)])

    def test_score(self):
        state = game_state.GameState()
        score = state.get_score()
        self.assertEqual(score, (0, 0))

    def test_playable_roads(self):
        self.assertEqual(len(game_state.playable_roads.keys()), 36)

        if GUI_ENABLED:
            button_str = "Draw Adjacent Roads"
            window = gui.create_window()
            draw_roads_event_loop(window, button_str)

    def test_get_playable_nodes(self):
        roads = []
        nodes = game_state.get_playable_nodes(roads)
        expected_nodes = []
        self.assertEqual(nodes, expected_nodes)

        roads = [(2.5, 2), (3, 2.5)]
        nodes = game_state.get_playable_nodes(roads)
        expected_nodes = [(2, 2), (3, 2), (3, 3)]
        self.assertEqual(nodes, expected_nodes)

        roads = [(0, 2.5), (0.5, 2), (0.5, 3), (1, 2.5)]
        nodes = game_state.get_playable_nodes(roads)
        expected_nodes = [(0, 2), (0, 3), (1, 2), (1, 3)]
        self.assertEqual(nodes, expected_nodes)


if __name__ == "__main__":
    unittest.main()
