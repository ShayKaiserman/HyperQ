import numpy as np
from enum import Enum


class Similarity(Enum):
    likelihood = 1
    likelihood_scaled = 2
    cosine = 3


class UltraQLearningAgent:
    def __init__(self, n_states, n_actions, n_opponent_actions, alpha=0.01, gamma=0.9, mu=0.01,
                 similarity=Similarity.cosine, epsilon=0.1, prior_strategy=None, init=None):
        """
        Initializes the Ultra-Q learning agent for a single-state game.

        Parameters:
        - n_actions: Number of actions available to the agent.
        - n_opponent_actions: Number of actions available to the opponent.
        - alpha: Learning rate for Q-value updates. Controls how quickly the agent incorporates new information.
        - gamma: Discount factor for future rewards. Determines the importance of future rewards.
        - mu: Learning rate for posterior updates (EMA). Controls how quickly the agent updates its beliefs about the opponent's strategy.
        - similarity: Method to calculate similarity between strategies. Options are:
            - Similarity.likelihood: Uses the probability of the chosen action.
            - Similarity.likelihood_scaled: Scales the likelihood by the actual strategy.
            - Similarity.cosine: Uses cosine similarity between strategies.
        - epsilon: Exploration rate for epsilon-greedy policy. Controls the trade-off between exploration and exploitation.
        - init: Initial value for Q-table entries. If None, random initialization is used.
        """

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_opponent_actions = n_opponent_actions
        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu
        self.epsilon = epsilon
        self.similarity = similarity

        # Initialize Q-table
        if init is not None:
            self.Q = np.full((n_states, n_opponent_actions, n_actions), init)
        else:
            self.Q = np.random.random_sample((n_states, n_opponent_actions, n_actions))

        # Initialize agent's strategy and posterior
        if prior_strategy is not None:
            self.strategy = prior_strategy
            for state in range(n_states):
                for a in range(n_actions):
                    self.Q[state, :, a] = np.log(self.strategy[state, a]+0.1)  # Use log to set initial Q-values
        else:
            # Random probabilities
            self.strategy = np.random.dirichlet(np.ones(n_actions), size=n_states)
            # # Uniformly distributed strategy
            # self.strategy = np.ones((n_states, n_actions)) / n_actions
        self.posterior = np.ones(n_opponent_actions) / n_opponent_actions

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Choose an action based on our strategy
            return np.random.choice(self.n_actions, p=self.strategy[state])
            # # Do the most probable action in the given state, based on our learned strategy
            # return np.argmax(self.strategy[state])

    def update_posterior(self, opponent_action):
        self.posterior *= (1 - self.mu)
        self.posterior[opponent_action] += self.mu
        self.posterior /= self.posterior.sum()

    def calculate_similarity(self, x, x_real, action):
        if self.similarity == Similarity.likelihood:
            return x[action]
        elif self.similarity == Similarity.likelihood_scaled:
            return x[action] * x_real[action]
        elif self.similarity == Similarity.cosine:
            return np.dot(x, x_real) / (np.linalg.norm(x) * np.linalg.norm(x_real))

    def index_to_strategy(self, index):
        strategy = np.zeros(self.n_actions)
        strategy[index] = 1
        return strategy

    def update_Q(self, state, action, reward, next_state, opponent_action, opponent_strategy):
        """
        Updates the Q-table using the observed reward and transition.
        Contrary to Hyper-Q, it's update the entire Q-table at once with each iteration.
        """

        self.update_posterior(opponent_action)

        max_next_q = np.max(self.Q[next_state])

        x_real = self.strategy[state]  # The agent's current strategy for this state

        for x_index in range(self.n_actions):
            for y_index in range(self.n_opponent_actions):
                x = self.index_to_strategy(x_index)  # Convert index to a strategy
                s = self.calculate_similarity(x, x_real, action)

                bellman_error = s * self.posterior[y_index] * (
                        reward + self.gamma * max_next_q - self.Q[state, y_index, x_index]
                )
                self.Q[state, y_index, x_index] += self.alpha * bellman_error

    def update_strategy(self, state, action, reward, next_state, opponent_action, opponent_strategy=None):
        """
        Updates the strategy (policy) table based on the current Q-values.

        Parameters:
        - state: Current state.
        """

        # Update the Q-values
        self.update_Q(state, action, reward, next_state, opponent_action, opponent_strategy)

        # Update the Strategy accordingly
        exp_Q = np.exp(self.Q[state, :, :])  # Exponential of Q-values for softmax
        self.strategy[state] = exp_Q.sum(axis=0) / exp_Q.sum()  # Normalize to get probabilities
