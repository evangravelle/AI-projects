import game_state
import unittest

class TestGameState(unittest.TestCase):

    def game_state(self):
        state = game_state.GameState()
        state.update([])
        self.assertEqual(state.nodes, [])
        self.assertEqual(state.roads, [])
        state.update([(0,2), ((0,2), (0,3))])
        self.assertEqual(state.nodes, [(0,2), ])
        self.assertEqual(state.roads, [((0,2), (0,3))])
        self.assertRaises(state.update([(0,2), ((1,1), (1,2), (1,3))]))


    def test2(self):
        self.assertEqual(1, 1.0)

if __name__ == '__main__':
    unittest.main()
