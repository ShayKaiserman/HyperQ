import numpy as np


class IGAAgent:
    def __init__(self, n_states, n_actions, eta=0.01, prior_strategy=None):
        """
        Initializes an IGA agent.

        Parameters:
        - n_states: Number of states in the environment.
        - n_actions: Number of actions available to the agent.
        - eta: Learning rate for policy update.
        - prior_strategy: Initial strategy to start from.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = eta

        # Initialize the agent's strategy
        if prior_strategy is not None:
            self.strategy = prior_strategy
        else:
            # Random probabilities
            self.strategy = np.random.dirichlet(np.ones(n_actions), size=n_states)
            # uniform distributed
            # self.strategy = np.ones((n_states, n_actions)) / n_actions

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def select_action(self, state):
        return np.random.choice(self.n_actions, p=self.strategy[state])

    def update_strategy(self, state, action, reward, next_state=None, opponent_action=None, opponent_strategy=None):
        """
        Updates the strategy using gradient ascent.

        Parameters:
        - state: Current state.
        - action: Action taken by the agent.
        - reward: Reward received.
        - next_state (not necessary) : Next state after taking the action.
        - opponent_action (not necessary): Action taken by the opponent.
        - opponent_strategy (not necessary): Estimated strategy of the opponent.
        """

        # The payoff matrix for rock-paper-scissors game
        R = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])

        # # The payoff matrix for hill-climbing (expected rewards)
        # R = np.array([
        #     [1, - 40, - 10.],
        #     [-40, - 3, - 4.],
        #     [-7.9, - 10, - 5.]
        # ])

        # Compute the gradient of the policy
        gradient = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            # gradient[a] = reward if a == action else 0
            gradient[a] = R[a, :].dot(opponent_strategy[state])

        self.strategy[state] += self.eta * gradient

        # Clip probabilities to [0, 1] and normalize
        self.strategy = np.clip(self.strategy, 0, 1)
        self.strategy /= self.strategy.sum(axis=1, keepdims=True)

        # # Compute the gradient of the policy
        # gradient = np.zeros(self.n_actions)
        # for a in range(self.n_actions):
        #     gradient[a] = reward * (1 if a == action else 0 - self.strategy[state, a])
        #
        # # Update the policy
        # self.strategy[state] += self.alpha * gradient
        # self.strategy[state] = np.clip(self.strategy[state], 0, 1)
        # self.strategy[state] /= self.strategy[state].sum()  # Ensure it's a valid probability distribution
