import os
import numpy as np
import matplotlib.pyplot as plt


def clean_and_plot(array: np.ndarray, keys: list, save_path: str, title: str = None):
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
        plt.plot(cleaned_array[:, i], label=key)

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

    output_dir = "/home/shayka/Projects/MultiAgent-learning/HyperQ/results/HyperQ_bayesian_Vs_IGA"

    strategies = {
        "student": f"{output_dir}/student_strategies.npy",
        "teacher": f"{output_dir}/teacher_strategies.npy",
        "teacher's estimated": f"{output_dir}/teacher_est_strategies.npy"
    }

    for strategy_name, file in strategies.items():
        print(strategy_name)
        # Load
        strategy = np.load(file)
        # Plot
        actions_names = [f"action {i}" for i in range(strategy.shape[1])]
        clean_and_plot(array=strategy, keys=actions_names, title=f"{strategy_name} strategy",
                       save_path=os.path.join(output_dir, f"{strategy_name}_strategies.png"))




if __name__ == "__main__":
    main()