import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from same_action_agent import ConstantAgent
from random_agent import RandomAgent
from iga import IGAAgent
from phc import PHCAgent
from hyper_q import HyperQLearningAgent
from ultra_q import UltraQLearningAgent, Similarity

class Agent:
    def __init__(self, agent_type: str = '',
                 n_states: int = None, n_actions: int = None, n_opponent_actions: int = None,
                 estimator: str = 'bayesian',
                 initial_strategy: np.ndarray = None):

        self.n_states = n_states
        self.n_actions = n_actions

        if agent_type == 'Constant':
            self.agent = ConstantAgent(n_states=n_states, n_actions=n_actions)
        elif agent_type == 'Random':
            self.agent = RandomAgent(n_states=n_states, n_actions=n_actions)
        elif agent_type == 'IGA':
            self.agent = IGAAgent(n_states=n_states, n_actions=n_actions, prior_strategy=initial_strategy)
        elif agent_type == 'PHC':
            self.agent = PHCAgent(n_states=n_states, n_actions=n_actions, prior_strategy=initial_strategy)
        elif agent_type == 'HyperQ':
            self.agent = HyperQLearningAgent(n_states=n_states, n_actions=n_actions,
                                             n_opponent_actions=n_opponent_actions, estimator=estimator,
                                             prior_strategy=initial_strategy)
        elif agent_type == 'UltraQ':
            self.agent = UltraQLearningAgent(n_states=n_states, n_actions=n_actions,
                                             n_opponent_actions=n_opponent_actions,
                                             prior_strategy=initial_strategy, similarity=Similarity.cosine)
            pass
        else:
            raise KeyError(f"Agent of type {agent_type} doesn't defined")

        self.agent_name = f'{agent_type}_{estimator}' if agent_type == 'HyperQ' else agent_type
        self.total_rewards = 0

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def select_action(self, state):
        return self.agent.select_action(state)

    def update_strategy(self, state, action, reward, next_state, opponent_action, opponent_strategy=None):
        self.agent.update_strategy(state, action, reward, next_state, opponent_action, opponent_strategy)
        self.total_rewards += reward

    def get_strategy(self):
        return self.agent.strategy

    def random_strategy_restarts(self):
        if self.agent_name not in ['Constant', 'Random']:
            self.agent.strategy = np.random.dirichlet(np.ones(self.agent.n_actions), size=self.agent.n_states)

    def uniform_strategy(self):
        # Uniformly distributed strategy
        self.agent.strategy = np.ones((self.agent.n_states, self.agent.n_actions)) / self.agent.n_actions
