import numpy as np


class HyperQLearningAgent:
    def __init__(self, n_states, n_actions, n_opponent_actions, alpha=0.01, gamma=0.9, epsilon=0.1, estimator='bayesian',
                 decay=0.0005, prior_strategy=None):
        """
        Initializes the Hyper-Q learning agent.

        Parameters:
        - n_states: Number of states in the environment.
        - n_actions: Number of actions available to the agent.
        - n_opponent_actions: Number of actions available to the opponent.
        - alpha: Learning rate.
        - gamma: Discount factor.
        - epsilon: Exploration rate for epsilon-greedy policy. Controls the trade-off between exploration and exploitation.
        - estimator: Method to estimate opponent strategies ('omniscient', 'EMA', 'bayesian').
        - decay: Decay rate for EMA.
        - prior_strategy: Initial strategy to start from.
        """

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_opponent_actions = n_opponent_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        # self.Q = np.zeros((n_states, n_opponent_actions, n_actions))
        self.Q = np.random.random_sample((n_states, n_opponent_actions, n_actions))
        self.bellman_errors = []

        # Initialize the agent's strategy
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

        # Initialize the estimator based on the specified type
        if estimator == 'omniscient':
            self.estimator = OmniscientEstimator(n_actions)
        elif estimator == 'ema':
            self.estimator = EMAEstimator(n_actions, weighted_multiplier=decay)
        elif estimator == 'bayesian':
            self.estimator = BayesianEstimator(n_actions)
        else:
            raise ValueError("Invalid estimator type. Choose 'omniscient', 'ema', or 'bayesian'.")
        self.estimator_type = estimator

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

    def update_Q(self, state, action, reward, next_state, opponent_action, opponent_strategy):
        """
        Updates the Q-table using the observed reward and transition.

        Parameters:
        - state: Current state.
        - action: Action taken by the agent.
        - reward: Reward received.
        - next_state: Next state after taking the action.
        - opponent_action: Opponent action in the current state.
        - opponent_strategy: Opponent strategy (for all states)
        """

        # First get the opponent's strategy estimation for this state ant the next one
        if self.estimator_type == 'omniscient':
            est_opponent_strategy = opponent_strategy[state]
            est_next_opponent_strategy = opponent_strategy[next_state]
        else:
            # self.estimator.update(opponent_action)
            est_opponent_strategy = self.estimator.get_estimated_strategy()
            self.estimator.update(opponent_action)
            est_next_opponent_strategy = self.estimator.get_estimated_strategy()

        # Update the Q-table based on the extended Bellman Equation
        current_q = self.Q[state, np.argmax(est_opponent_strategy), action]
        max_next_q = np.max(self.Q[next_state, np.argmax(est_next_opponent_strategy), :])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.Q[state, np.argmax(est_opponent_strategy), action]
        new_q = current_q + self.alpha * td_error
        self.Q[state, np.argmax(est_opponent_strategy), action] = new_q
        self.bellman_errors.append(td_error) # Bellman error

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


class OmniscientEstimator:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.true_strategy = np.ones(n_actions) / n_actions

    def update(self, action):
        # In a real scenario, this would be updated with the true opponent strategy
        pass

    def get_estimated_strategy(self):
        return self.true_strategy

    def set_true_strategy(self, strategy):
        self.true_strategy = strategy


class EMAEstimator:
    def __init__(self, n_actions, weighted_multiplier=0.005):
        self.n_actions = n_actions
        self.weighted_multiplier = weighted_multiplier
        self.strategy_estimate = np.ones(n_actions) / n_actions

    def update(self, action):
        action_vector = np.zeros(self.n_actions)
        action_vector[action] = 1
        self.strategy_estimate = ((1 - self.weighted_multiplier) * self.strategy_estimate
                                  + self.weighted_multiplier * action_vector)

    def get_estimated_strategy(self):
        return self.strategy_estimate


class BayesianEstimator:
    def __init__(self, n_actions, prior_counts=None):
        self.n_actions = n_actions
        if prior_counts is None:
            # starts with uniform prior
            self.counts = np.ones(n_actions)
        else:
            self.counts = prior_counts
        self.strategy_estimate = self.counts / np.sum(self.counts)

    def update(self, action):
        self.counts[action] += 1
        self.strategy_estimate = self.counts / np.sum(self.counts)

    def get_estimated_strategy(self):
        return self.strategy_estimate
