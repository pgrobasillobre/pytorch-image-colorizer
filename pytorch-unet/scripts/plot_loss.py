import pickle
import matplotlib.pyplot as plt
import argparse
import os

def load_history(path):
    with open(path, "rb") as f:
        history = pickle.load(f)
    return history

def plot_loss(history):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Over Epochs")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
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
