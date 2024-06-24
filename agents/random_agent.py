import numpy as np

class RandomAgent:
    def __init__(self, n_states, n_actions):
        """
        Initializes A Random player.

        Parameters:
        - n_states: Number of states in the environment.
        - n_actions: Number of actions available to the agent.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        # Uniform distribution strategy
        self.strategy = np.ones((n_states, n_actions)) / n_actions

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def select_action(self, state):
        return np.random.randint(self.n_actions)

    def update_strategy(self, state=None, action=None, reward=None, next_state=None, opponent_action=None,
                        opponent_strategy=None):
        pass

