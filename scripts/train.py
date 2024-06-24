import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np

from agents.general_agent import Agent
from enviroments.env_zoo import select_environment
from utils import *

import wandb
import pickle

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.


def save(episode, student_agent, student_strategies, teacher_strategies, teacher_est_strategies, save_path):
    with open(f'{save_path}/trained_student_agent.pkl', 'wb') as f:
        pickle.dump(student_agent, f)
    # save all the strategies
    np.save(f'{save_path}/student_strategies.npy', student_strategies)
    np.save(f'{save_path}/teacher_strategies.npy', teacher_strategies)
    np.save(f'{save_path}/teacher_est_strategies.npy', teacher_est_strategies)
    # Print current strategies
    print(f"Episode {episode}: Saved student agent.")
    print(f"Teacher Agent Strategy:\n{teacher_strategies[episode]}")
    print(f"Student Agent Strategy:\n{student_strategies[episode]}")


def train(student_agent, teacher_agent, env, n_episodes=10000, save_interval=1000, save_path=None):
    """
    Trains a student agent against a teacher agent in the given environment.

    Parameters:
    - student_agent: The student agent to be trained.
    - teacher_agent: The teacher agent.
    - env: The environment in which the agents interact.
    - n_episodes: Number of episodes to train.
    - save_interval: Interval at which the student agent is saved.
    """

    # Initialize W&B
    wandb.init(project="multi-agent-training", name=f"training-{student_agent.agent_name}_vs._{teacher_agent.agent_name}")

    # Tracking variables
    reward_diffs = []
    bellman_errors = []
    average_bellman_error = 0.0
    student_strategies = np.zeros((n_episodes, env.n_actions))
    teacher_strategies = np.zeros((n_episodes, env.n_actions))
    teacher_est_strategies = np.zeros((n_episodes, env.n_actions))

    for episode in range(n_episodes):
        state = env.initial_state

        # Each agent selects an action
        teacher_action = teacher_agent.select_action(state)
        student_action = student_agent.select_action(state)

        # Getting the rewards
        teacher_reward, student_reward, next_state = env.step(teacher_action, student_action)

        # Each agent updates their strategy accordingly
        teacher_strategy = teacher_agent.get_strategy()
        student_strategy = student_agent.get_strategy()
        teacher_agent.update_strategy(state, teacher_action, teacher_reward, next_state,
                                      opponent_action=student_action, opponent_strategy=student_strategy)
        student_agent.update_strategy(state, student_action, student_reward, next_state,
                                      opponent_action=teacher_action, opponent_strategy=teacher_strategy)

        # Store all the strategies
        teacher_strategies[episode] = teacher_strategy
        student_strategies[episode] = student_strategy
        opponent_estimator = student_agent.agent.get('estimator', None)
        if opponent_estimator:
            teacher_est_strategies[episode] = opponent_estimator.get_estimated_strategy()

        # Calculate the rewards diff and update tracking
        rewards_diff = student_reward - teacher_reward
        reward_diffs.append(rewards_diff)

        bellman_error = student_agent.agent.get('bellman_errors', [0])[-1]
        bellman_errors.append(bellman_error)
        average_bellman_error = 0.05 * bellman_error + 0.95 * average_bellman_error

        # Log metrics to W&B
        avg_reward_diff = np.mean(reward_diffs[-100:])  # Average reward diff over the last 100 episodes
        wandb.log({
            "Episode": episode,
            "Student Reward": student_reward,
            "Teacher Reward": teacher_reward,
            "Reward Diff": rewards_diff,
            "Average Reward Diff": avg_reward_diff,
            "Bellman Error": average_bellman_error,
            # "Student Strategy": wandb_display_list(episode, student_strategies,
            #                                        keys=[f'action {i}' for i in range(student_agent.agent.n_actions)],
            #                                        title='Student Strategy'),
            # "Teacher Strategy": wandb_display_list(episode, teacher_strategies,
            #                                        keys=[f'action {i}' for i in range(teacher_agent.agent.n_actions)],
            #                                        title='Teacher Strategy'),
        })

        # Print rewards diff for the current episode
        print(f"Episode {episode} Rewards: Student = {student_reward} | Teacher = {teacher_reward}"
              f" | Diff = {rewards_diff}")

        # Save the student agent at the specified interval
        if save_path and episode % save_interval == 0:
            save(episode, student_agent, student_strategies, teacher_strategies, teacher_est_strategies, save_path)

    save(episode, student_agent, student_strategies, teacher_strategies, teacher_est_strategies, save_path)
    print("Training completed.")

    # Ensure the W&B run is finished properly
    wandb.finish()


class ConditionalArgument(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, 'student') == 'HyperQ' and not values:
            parser.error("--estimator_type is required when student is 'HyperQ'.")
        setattr(namespace, self.dest, values)

def main():
    parser = argparse.ArgumentParser(description='Train an agent (student) against another agent (teacher)'
                                                 ' in a multi-agent environment.')
    parser.add_argument('--teacher', type=str, required=True, choices=['Constant', 'Random', 'IGA', 'PHC', 'HyperQ', 'UltraQ'],
                        help='Type of the first agent.')
    parser.add_argument('--student', type=str, required=True, choices=['Constant', 'Random', 'IGA', 'PHC', 'HyperQ', 'UltraQ'],
                        help='Type of the second agent.')
    parser.add_argument('--estimator_type', type=str, default='bayesian',
                        choices=['omniscient', 'EMA', 'bayesian'],
                        help='Estimator type for the second agent. (required only if the student is HyperQ)',
                        action=ConditionalArgument)
    parser.add_argument('--env', type=str, default='rock-paper-scissors', help='The game environment')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train.')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save agent every X episodes')
    parser.add_argument('--initial_student_strategy', type=parse_float_list, default=None,
                        help='comma-separated list of floats, representing probabilities of choosing different actions')

    args = parser.parse_args()

    # Login to wandb
    wandb.login()

    # Initialize train parameters
    num_episodes = args.episodes
    save_interval = args.save_interval
    teacher_name = args.teacher
    student_name = f'{args.student}_{args.estimator_type}' if args.student == 'HyperQ' else args.student
    save_path = f'../results/{student_name}_Vs_{teacher_name}'
    os.makedirs(save_path, exist_ok=True)

    # Initialize the environment and the agents
    env = select_environment(env_name=args.env)
    initial_student_strategy = np.array([args.initial_student_strategy]*env.n_states) if args.initial_student_strategy else None
    estimator_type = getattr(args, 'estimator_type')
    student = Agent(args.student, n_states=env.n_states, n_actions=env.n_actions, n_opponent_actions=env.n_actions,
                    estimator=estimator_type, initial_strategy=initial_student_strategy)
    teacher = Agent(args.teacher, n_states=env.n_states, n_actions=env.n_actions, n_opponent_actions=env.n_actions,
                    estimator=estimator_type)

    train(student, teacher, env, num_episodes, save_interval=save_interval, save_path=save_path)


if __name__ == "__main__":
    main()
