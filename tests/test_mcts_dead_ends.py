import math
import unittest

from rap.mcts import MCTS, MCTSNode


class Node(MCTSNode):
    def __init__(self, children=(), reward=1, terminal=False, visited=True):
        self.children = children
        self._reward = reward
        self._terminal = terminal
        self._visited = visited

    def find_children(self):
        return self.children

    def find_one_child(self):
        return self.children[0] if self.children else None

    @property
    def is_terminal(self):
        return self._terminal

    @property
    def reward(self):
        return self._reward

    @property
    def visited(self):
        return self._visited


class DeadEndTests(unittest.TestCase):
    def test_repeated_rollouts_on_expanded_dead_end(self):
        for prior in (False, True):
            with self.subTest(prior=prior):
                root = Node()
                search = MCTS(prior=prior)
                for _ in range(3):
                    search.rollout(root)
                self.assertEqual(search.N[root], 3)
                self.assertEqual(search.Q[root], 3)
                self.assertEqual(search.children[root], ())

    def test_repeated_rollouts_through_dead_end_child(self):
        child = Node(reward=2)
        root = Node(children=(child,))
        search = MCTS()
        for _ in range(3):
            search.rollout(root)
        self.assertEqual(search.N[root], 3)
        self.assertEqual(search.N[child], 3)
        self.assertEqual(search.Q[root], 9)
        self.assertEqual(search.Q[child], 6)

    def test_expanded_dead_end_has_no_terminal_reward(self):
        root = Node()
        search = MCTS()
        search._expand(root)
        self.assertEqual(search.max_terminal(root), (root, -math.inf))
        self.assertEqual(search.max_mean_terminal(root), (root, -math.inf))

    def test_valid_terminal_branch_wins_over_dead_end(self):
        dead_end = Node(reward=100)
        terminal = Node(reward=2, terminal=True)
        root = Node(children=(dead_end, terminal))
        search = MCTS()
        search._expand(root)
        search._expand(dead_end)
        self.assertEqual(search.max_terminal(root), (terminal, 3))

    def test_unexpanded_nonterminal_remains_ineligible(self):
        root = Node()
        self.assertEqual(MCTS().max_terminal(root), (root, -math.inf))

    def test_unvisited_terminal_remains_ineligible(self):
        terminal = Node(terminal=True, visited=False)
        self.assertEqual(MCTS().max_terminal(terminal), (terminal, -math.inf))


if __name__ == "__main__":
    unittest.main()
