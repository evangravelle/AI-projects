import game_state
import unittest

class TestGameState(unittest.TestCase):

    def test_game_state(self):
        state = game_state.GameState()
        state.update("p1", [])
        self.assertEqual(state.nodes["p1"], [])
        self.assertEqual(state.roads["p1"], [])
        state.update("p2", [(0,2), ((0,2), (0,3))])
        self.assertEqual(state.nodes["p2"], [(0,2), ])
        self.assertEqual(state.roads["p2"], [((0,2), (0,3))])
        with self.assertRaises(AssertionError):
            state.update("p1", [(0,2), ((1,1), (1,2), (1,3))])


    def test_score(self):
        state = game_state.GameState()
        score = state.get_score()
        self.assertEqual(score, (0,0))

    def test_playable_roads(self):
        self.assertEqual(len(game_state.playable_roads.keys()), 36)

if __name__ == '__main__':
    unittest.main()
