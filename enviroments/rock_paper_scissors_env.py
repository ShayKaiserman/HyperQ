import numpy as np


class RockPaperScissorsEnv:
    def __init__(self):
        self.name = 'rock-paper-scissors'
        self.type = 'competitive'
        self.initial_state = 0
        self.n_actions = 3
        self.n_states = 1  # Single state game
        self.action_space = ['Rock', 'Paper', 'Scissors']

    @staticmethod
    def step(action1, action2):
        payoff_matrix = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])

        reward1 = payoff_matrix[action1, action2]
        # because it's zero-sum game
        reward2 = -reward1

        # return reward1, reward2, 0, {}  # Rewards, next state (always 0), info dict
        return reward1, reward2, 0  # Rewards, next state (always 0)

    @staticmethod
    def reset():
        return 0 # Always return the single state