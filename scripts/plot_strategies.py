import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def clean_and_plot(array: np.ndarray, keys: list, save_path: str, title: str = None, log_scale: bool = False):
    """
    This function:
    1. Removes rows from the array where all values are zeros.
    2. Plots each column of the remaining array with a legend corresponding to the keys.
    3. Saves the plot to the specified save_path.

    Parameters:
    array (ndarray): Input array of shape (X, 3).
    keys (list of str): List of keys corresponding to the columns.
    save_path (str): Path to save the generated plot.

    Raises:
    ValueError: If the array does not have 3 columns or if the length of keys does not match the number of columns.
    """

    if len(keys) != array.shape[1]:
        raise ValueError("Length of keys must match the number of columns in the array.")

    # Step 1: Remove rows where all values are zeros
    cleaned_array = array[~np.all(array == 0, axis=1)]

    # Check if cleaned_array is empty
    if cleaned_array.size == 0:
        raise ValueError("The cleaned array has no rows after removing rows with all zeros.")

    # Step 2: Plot each column
    plt.figure()
    for i, key in enumerate(keys):
        # smoothed_plot = adaptive_savgol_filter(cleaned_array[:, i], max_window_length=2000, poly_order=2)
        # plt.plot(smoothed_plot, label=key)
        plt.plot(cleaned_array[:, i], label=key)

    if log_scale:
        plt.xscale('log')

    # Add legend
    plt.legend()

    # Add titles and labels
    if title:
        plt.title(title)

    # Step 3: Save the plot
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def main():

    # output_dir = "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/rock-paper-scissors/HyperQ_bayesian_Vs_HyperQ"
    # output_dir = "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/hill-climbing/HyperQ_bayesian_Vs_Constant"

    # dirs = [
    #     # "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/HyperQ_bayesian_Vs_Random",
    #     # "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/HyperQ_ema_Vs_Random",
    #     # '/home/shayka/Projects/MultiAgent-learning/HyperQ/results/HyperQ_omniscient_Vs_Random',
    #     '/home/shayka/Projects/MultiAgent-learning/HyperQ/results/rock-paper-scissors/HyperQ_bayesian_Vs_Constant-epsilon=0',
    #     '/home/shayka/Projects/MultiAgent-learning/HyperQ/results/rock-paper-scissors/HyperQ_bayesian_Vs_Constant-epsilon=0@after_2000_episodes',
    #     # '/home/shayka/Projects/MultiAgent-learning/HyperQ/results/PHC_Vs_Random'
    # ]

    log_scale = True

    main_dir = "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/rock-paper-scissors"
    # # main_dir = "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/rock-paper-scissors-smooth"
    # # main_dir = "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/hill-climbing"
    # # main_dir = "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/prisoners-dilemma"
    dirs = [os.path.join(main_dir, sub_dir) for sub_dir in os.listdir(main_dir)
            if os.path.isdir(os.path.join(main_dir, sub_dir))]

    for output_dir in dirs:

        print(f"{PRINT_START}{BLUE}{output_dir}{PRINT_STOP}")

        strategies = {
            "my": f"{output_dir}/student_strategies.npy",
            "opponent": f"{output_dir}/teacher_strategies.npy",
            "opponent's estimated": f"{output_dir}/teacher_est_strategies.npy"
        }

        for strategy_name, file in strategies.items():
            print(strategy_name)
            # Load
            strategy = np.load(file)
            # Plot
            try:
                # actions_names = [f"action {i}" for i in range(strategy.shape[1])]
                actions_names = ["rock", "paper", "scissors"]
                plot_name = f"{strategy_name}_strategies-log_scale.png" if log_scale else f"{strategy_name}_strategies.png"
                clean_and_plot(array=strategy, keys=actions_names, title=f"{strategy_name} strategy",
                               save_path=os.path.join(output_dir, plot_name),
                               log_scale=log_scale)
            except:
                continue




if __name__ == "__main__":
    main()
