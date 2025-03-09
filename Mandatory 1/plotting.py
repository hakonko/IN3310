import matplotlib
matplotlib.use('Agg') # works for saving on none-GUI systems
import matplotlib.pyplot as plt

import pandas as pd

def plot_map_per_class(classes, class_accs, map_scores, base_path, file_name, figsize=(10, 5)):

    save_path = base_path / file_name

    plt.figure(figsize=figsize)

    # plotting accuracy per class
    for i, cls in enumerate(classes):
        acc_per_class = [epoch[i] for epoch in class_accs]
        plt.plot(range(1, len(acc_per_class) + 1), acc_per_class, linestyle='-', linewidth=1, label=f"{cls}")

    # plotting mAP
    plt.plot(range(1, len(map_scores) + 1), map_scores, linestyle='-', linewidth=3, color='black', label="mAP")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("mAP and Mean Accuracy per Class per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_losses(file_path, file_name, train_losses, val_losses=None, figsize=(10, 5), labels=["Train losses", "Validation losses"], title="Train/Validation Loss Plot"):

    save_path = file_path / file_name

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, label=labels[0])
    if val_losses:
        plt.plot(epochs, val_losses, label=labels[1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
