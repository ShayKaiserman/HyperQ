# HyperQ
Implementation of Hyper-Q agent in multiagents framework 

The original paper:

[Extending Q-Learning to General Adaptive Multi-Agent Systems](assests/L 4%20-%20NIPS-2003-extending-q-learning-to-general-adaptive-multi-agent-systems-Paper.pdf)

Another paper explains further extension to Ultra-Q agent:

[Efficient Bayesian Ultra-Q Learning for Multi-Agent Games](assests/Bayesian_Ultra_Q.pdf)

---

## Usage:

1. First, to keep track on multiple experiments and see in live the learning process - you'll need to configure .env 
    file with WANDB_API_KEY, see this example .evn file: [.env_example](.env_example)

    To further understand how Weights&Biases (WANDB) works and how to create the API key: https://docs.wandb.ai/quickstart 

2.  To test agents against each other you need to use [train.py](scripts/train.py).

    `train.py` is a script designed to 'train' an agent (student) against another agent (teacher) in various multi-agent environments. 

- ### Required Arguments
  - `--teacher`: Specifies the type of the teacher agent.
    - Choices: `Constant`, `Random`, `IGA`, `PHC`, `HyperQ`, `UltraQ`

  - `--student`: Specifies the type of the student agent.
    - Choices: `Constant`, `Random`, `IGA`, `PHC`, `HyperQ`, `UltraQ`

- ### Optional Arguments

  - `--estimator_type`: Defines the estimator type for the student agent (only required if the student is `HyperQ`).
    - Default: `bayesian`
    - Choices: `omniscient`, `ema`, `bayesian`

  - `--env`: Sets the game environment for training.
    - Default: `rock-paper-scissors`
    - Choices: `rock-paper-scissors`, `hill-climbing`, `prisoners-dilemma`

  - `--episodes`: Determines the number of episodes to train.
    - Default: 10000

  - `--save_interval`: Specifies the interval for saving the agent's state.
    - Default: 1000

  - `--random_restart_interval`: Indicates the number of time steps between random strategy restarts (0 or negative means no restarts).
    - Default: 1000

  - `--initial_student_strategy`: A comma-separated list of floats representing the initial probabilities of the student choosing different actions.

  - `--start_from_random`: A flag to indicate whether to start from a uniform strategy.

- ### Example Command

  ```bash
  python train.py --teacher Random --student HyperQ --estimator_type bayesian --env rock-paper-scissors --episodes 5000 --save_interval 500 --random_restart_interval 200 --initial_student_strategy [0.5,0.2,0.3]
  ```

  This command sets the teacher to a Random agent and the student to a HyperQ agent with a bayesian estimator,
  running in the rock-paper-scissors environment for 5000 episodes, saving every 500 episodes,
  restarting randomly every 200 steps, with an initial strategy of [0.3, 0.3, 0.4].

3. After the match, you can plot the actions/ estimated actions of the two agents in graphs with
[plot_strategies.py](scripts/plot_strategies.py)

---

## More details:

Our presentation: [Extending Q-Learning to General Adaptive Multi-Agent Systems.pptx](assests/Extending%20Q-Learning%20to%20General%20Adaptive%20Multi-Agent%20Systems.pptx)

Detailed Explanations with examples to EMA and Bayesian opponent's strategy
estimation techniques: [Examples for Hyper Q.pdf](assests/Examples%20for%20Hyper%20Q.pdf)