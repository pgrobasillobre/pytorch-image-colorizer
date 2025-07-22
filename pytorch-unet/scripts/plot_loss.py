import pickle
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def load_history(path):
    """
    Load training history from a pickle (.pkl) file.

    Args:
        path (str): Path to the pickle file containing training history.

    Returns:
        dict: Dictionary containing training and validation loss history.
    """
    with open(path, "rb") as f:
        history = pickle.load(f)
    return history

def plot_loss(history):
    """
    Plot training and validation loss curves.

    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' lists.
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    fontsize_tics   = 18
    fontsize_labels = 20
    fontsize_titles = 22

    # Font settings for Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['mathtext.sf'] = 'Times New Roman'
    plt.rcParams['mathtext.default'] = 'regular'

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot lines
    ax.plot(range(len(train_loss)), train_loss, marker='^', label="Train Loss", linewidth=2, markersize=10)
    ax.plot(range(len(val_loss)), val_loss, marker='o', label="Validation Loss", linewidth=2, markersize=10)


    # Labels and title
    ax.set_xlabel("Epoch", fontsize=fontsize_labels, labelpad=10)
    ax.set_ylabel("Loss (MSE)", fontsize=fontsize_labels)
    ax.set_title("Loss Over Epochs", fontsize=fontsize_titles, pad=20)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=fontsize_tics)
    ax.set_xticks(range(0, len(train_loss) + 1, 2))


    # Add grid, legend, layout
    ax.grid(False)
    ax.legend(fontsize=fontsize_tics)
    
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15)  # Values < 1 move the plot down to create more top margin
    plt.show()

def main():
    """
    Parse command-line arguments and plot training history from a specified pickle file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default="pytorch/models/colorizer_training_history.pkl",
                        help="Path to training history .pkl file")
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"Error: File not found: {args.history}")
        return

    history = load_history(args.history)
    plot_loss(history)

if __name__ == "__main__":
    main()
