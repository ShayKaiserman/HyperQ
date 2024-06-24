import numpy as np

class ConstantAgent:
    def __init__(self, n_states, n_actions):
        """
        Initializes A player that his strategy is always to do the same action.

        Parameters:
        - n_states: Number of states in the environment.
        - n_actions: Number of actions available to the agent.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        # Strategy - choose always action 0 (with probability 1)
        self.strategy = np.zeros((n_states, n_actions))
        self.strategy[:, 0] = 1

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def select_action(self, state):
        return np.argmax(self.strategy[state, :])

    def update_strategy(self, state=None, action=None, reward=None, next_state=None, opponent_action=None,
                        opponent_strategy=None):
        pass

