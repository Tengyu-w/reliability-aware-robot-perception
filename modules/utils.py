# utils.py
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    """Set common random seeds for more reproducible prototype runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_training_history(history, filename='training_history.png'):
    """Plot training history."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_predictions(predictions, video_files, idx_to_class, filename='test_predictions.csv'):
    """Save prediction results."""
    predicted_indices = [p + 1 for p in predictions]

    results = pd.DataFrame({
        'video_file': video_files,
        'predicted_class': [idx_to_class[p] for p in predictions],
        'predicted_index': predicted_indices
    })

    results.to_csv(filename, index=False)
    print(f"Prediction results saved to '{filename}'")
    return results
