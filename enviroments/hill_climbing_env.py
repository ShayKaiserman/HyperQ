import numpy as np


class HillClimbingEnv:
    def __init__(self):
        self.name = 'hill-climbing'
        self.type = 'cooperative'
        self.initial_state = 0
        self.n_actions = 3
        self.n_states = 1  # Single state game
        self.action_space = ['Action1', 'Action2', 'Action3']

        self.probability_matrix, self.reward_matrix1, self.reward_matrix2 = self.initialize_game_rewards()

    @staticmethod
    def initialize_game_rewards():
        probability_matrix = np.array([
            [0.4, 0.25, 0.6],
            [0.25, 0.8, 0.8],
            [0.7, 0.6, 0.8]
        ])

        reward_matrix1 = np.array([
            [-3.5, -46, -6],
            [-46, -5, -5],
            [-4, -6, -6]
        ])

        reward_matrix2 = np.array([
            [4, -38, -16],
            [-38, 5, 0],
            [-17, -16, -1]
        ])

        return probability_matrix, reward_matrix1, reward_matrix2

    def step(self, action1, action2):
        probability = self.probability_matrix[action1, action2]

        expected_reward = (probability * self.reward_matrix1[action1, action2]
                           + (1-probability) * self.reward_matrix2[action1, action2])

        # Both players receive the same reward in this game
        return expected_reward, expected_reward, 0  # Rewards for both players, next state (always 0)

    @staticmethod
    def reset():
        return 0  # Always return the single state