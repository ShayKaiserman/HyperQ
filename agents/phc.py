import numpy as np

class PHCAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, delta=0.01, gamma=0.9, epsilon=0.1, prior_strategy=None):
        """
        Initializes a PHC agent.

        Parameters:
        - n_states: Number of states in the environment.
        - n_actions: Number of actions available to the agent.
        - alpha: Learning rate for Q-value updates.
        - delta: Learning rate for policy update.
        - gamma: Discount factor.
        - epsilon: Exploration rate for epsilon-greedy policy.
        - prior_strategy: Initial strategy to start from.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table and agent's strategy
        # self.Q = np.zeros((n_states, n_actions))
        self.Q = np.random.random_sample((n_states, n_actions))

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
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.random.choice(self.n_actions, p=self.strategy[state])

    def update_Q(self, state, action, reward, next_state):
        best_next_q = np.max(self.Q[next_state])
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + \
                                self.alpha * (reward + self.gamma * best_next_q)

    def update_strategy(self, state, action, reward, next_state, opponent_action=None, opponent_strategy=None):
        """
        Updates the strategy using hill climbing and Q-value updates.

        Parameters:
        - state: Current state.
        - action: Action taken by the agent.
        - reward: Reward received.
        - next_state: Next state after taking the action.
        - opponent_action (not necessary): Action taken by the opponent.
        - opponent_strategy (not necessary): Estimated strategy of the opponent.
        """

        # Update Q-value
        self.update_Q(state, action, reward, next_state)

        # Update policy
        best_action = np.argmax(self.Q[state])
        update = np.where(np.arange(self.n_actions) == best_action, self.delta, -self.delta / (self.n_actions - 1))
        self.strategy[state] += update

        # Clip probabilities to [0, 1] and normalize
        self.strategy = np.clip(self.strategy, 0, 1)
        self.strategy /= self.strategy.sum(axis=1, keepdims=True)
