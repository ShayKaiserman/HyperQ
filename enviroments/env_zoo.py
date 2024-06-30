import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rock_paper_scissors_env import RockPaperScissorsEnv
from hill_climbing_env import HillClimbingEnv
from prisoners_dillema_env import PrisonersDilemmaEnv


def select_environment(env_name='rock-paper-scissors'):
    if env_name == 'rock-paper-scissors':
        return RockPaperScissorsEnv()
    elif env_name == 'prisoners-dilemma':
        return PrisonersDilemmaEnv()
    elif env_name == 'hill-climbing':
        return HillClimbingEnv()
    else:
        raise KeyError(f'{env_name} environment is not supported')

