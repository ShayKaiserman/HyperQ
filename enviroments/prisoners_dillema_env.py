import numpy as np

class PrisonersDilemmaEnv:
    def __init__(self):
        self.name = 'prisoners-dilemma'
        self.type = 'cooperative'
        self.initial_state = 0
        self.n_actions = 2
        self.n_states = 1  # Single state game
        self.action_space = ['Cooperate', 'Defect']

    @staticmethod
    def step(action1, action2):
        payoff_matrix = np.array([
            [-1, -3],  # Cooperate
            [0, -2]    # Defect
        ])

        reward1 = payoff_matrix[action1, action2]
        reward2 = payoff_matrix[action2, action1]

        return reward1, reward2, 0  # Rewards, next state (always 0)

    @staticmethod
    def reset():
        return 0  # Always return the single state