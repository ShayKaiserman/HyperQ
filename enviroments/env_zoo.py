import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rock_paper_scissors_env import RockPaperScissorsEnv


def select_environment(env_name='rock-paper-scissors'):
    if env_name == 'rock-paper-scissors':
        return RockPaperScissorsEnv()
    else:
        raise KeyError(f'{env_name} environment is not supported')

