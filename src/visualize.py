"""Visualization utilities for exploratory analysis and results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import os


def plot_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        model_name: str,
        save_path: Optional[str] = None,
        show: bool = True,
) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-o", label="Training Loss", linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, "r-o", label="Validation Loss", linewidth=2, markersize=6)

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Loss", fontsize=12, fontweight="bold")
    plt.title(f"{model_name} - Training Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(epochs)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: List[str],
        model_name: str,
        save_path: Optional[str] = None,
        show: bool = True,
        cmap: str = "Blues",
) -> None:
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )

    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
        models_metrics: Dict[str, Dict[str, float]],
        metric_name: str,
        save_path: Optional[str] = None,
        show: bool = True,
) -> None:
    model_names = list(models_metrics.keys())
    metric_values = [models_metrics[model][metric_name] for model in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values, color="steelblue", edgecolor="black", linewidth=1.2)

    plt.xlabel("Model", fontsize=12, fontweight="bold")
    plt.ylabel(metric_name.capitalize(), fontsize=12, fontweight="bold")
    plt.title(f"Model Comparison - {metric_name.capitalize()}", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    train_losses = [0.69, 0.58, 0.45, 0.38, 0.32, 0.28]
    val_losses = [0.72, 0.65, 0.55, 0.50, 0.48, 0.52]

    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="LSTMClassifier",
        save_path="outputs/figures/lstm_training_curves.png",
        show=True,
    )

    confusion_matrix = np.array([[45, 5], [8, 42]])
    class_names = ["Negative", "Positive"]

    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        model_name="LSTMClassifier",
        save_path="outputs/figures/lstm_confusion_matrix.png",
        show=True,
    )

    models_metrics = {
        "DenseNN": {"accuracy": 0.745, "f1": 0.749, "kappa": 0.490},
        "VanillaRNN": {"accuracy": 0.550, "f1": 0.622, "kappa": 0.100},
        "LSTM (Tuned)": {"accuracy": 0.525, "f1": 0.582, "kappa": 0.050},
    }

    plot_model_comparison(
        models_metrics=models_metrics,
        metric_name="accuracy",
        save_path="outputs/figures/model_comparison_accuracy.png",
        show=True,
    )
