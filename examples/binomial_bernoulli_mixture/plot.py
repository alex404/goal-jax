"""Plotting for Binomial-Bernoulli Mixture example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..shared import apply_style, example_paths, model_colors
from .types import BinomialBernoulliResults


def plot_training_elbo(
    ax: Axes,
    elbos: NDArray[np.float64],
) -> None:
    """Plot ELBO training curve."""
    ax.plot(elbos, color=model_colors[0])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("ELBO")
    ax.set_title("Training ELBO")
    ax.grid(True, alpha=0.3)


def plot_cluster_distribution(
    ax: Axes,
    predicted: NDArray[np.int64],
    true: NDArray[np.int64],
    n_clusters: int,
) -> None:
    """Plot predicted vs true cluster distribution."""
    x = np.arange(n_clusters)
    width = 0.35

    pred_counts = [np.sum(predicted == k) for k in range(n_clusters)]
    true_counts = [np.sum(true == k) for k in range(n_clusters)]

    ax.bar(x - width / 2, pred_counts, width, label="Predicted", color=model_colors[0])
    ax.bar(x + width / 2, true_counts, width, label="True", color=model_colors[1])

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Distribution")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


def plot_confusion_matrix(
    ax: Axes,
    predicted: NDArray[np.int64],
    true: NDArray[np.int64],
    n_clusters: int,
) -> None:
    """Plot confusion matrix between predicted and true clusters."""
    # Build confusion matrix
    conf_matrix = np.zeros((n_clusters, n_clusters))
    for pred, true_label in zip(predicted, true):
        conf_matrix[true_label, pred] += 1

    # Normalize by true label count
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    conf_matrix_norm = conf_matrix / row_sums

    im = ax.imshow(conf_matrix_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Cluster")
    ax.set_title("Confusion Matrix (normalized)")
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))

    # Add text annotations
    for i in range(n_clusters):
        for j in range(n_clusters):
            val = conf_matrix_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_reconstructions(
    ax: Axes,
    original: NDArray[np.float64],
    reconstructed: NDArray[np.float64],
    n_samples: int = 6,
    n_trials: int = 8,
) -> None:
    """Plot original vs reconstructed samples as heatmaps.

    Shows pairs of (original, reconstruction) for n_samples samples.
    Assumes 16-dimensional data (4x4 grid).
    """
    n_samples = min(n_samples, original.shape[0])

    # Create a grid: 2 rows (orig/recon) x n_samples columns
    grid = np.zeros((2 * 4 + 1, n_samples * 4 + (n_samples - 1)))  # +1 for separator

    for i in range(n_samples):
        orig_img = original[i].reshape(4, 4)
        recon_img = reconstructed[i].reshape(4, 4)

        col_start = i * 5  # 4 pixels + 1 gap
        # Original on top
        grid[0:4, col_start : col_start + 4] = orig_img
        # Reconstruction on bottom
        grid[5:9, col_start : col_start + 4] = recon_img

    ax.imshow(grid, cmap="viridis", vmin=0, vmax=n_trials)
    ax.set_title("Reconstructions (top: original, bottom: reconstructed)")
    ax.axis("off")


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(BinomialBernoulliResults, paths.load_analysis())

    # Extract data
    elbos = np.array(results["training_elbos"])
    predicted = np.array(results["predicted_labels"])
    true = np.array(results["test_labels"])
    test_data = np.array(results["test_data"])
    reconstructions = np.array(results["reconstructions"])

    n_clusters = len(results["cluster_distribution"])
    purity = results["cluster_purity"]
    recon_error = results["reconstruction_error"]

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax_elbo = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[0, 2])
    ax_recon = fig.add_subplot(gs[1, :])

    # Plot training ELBO
    plot_training_elbo(ax_elbo, elbos)

    # Plot cluster distribution
    plot_cluster_distribution(ax_dist, predicted, true, n_clusters)

    # Plot confusion matrix
    plot_confusion_matrix(ax_conf, predicted, true, n_clusters)

    # Plot reconstructions
    plot_reconstructions(ax_recon, test_data, reconstructions, n_samples=8)

    # Add summary text
    fig.suptitle(
        f"Binomial-Bernoulli Mixture: Purity={100 * purity:.1f}%, Recon Error={recon_error:.4f}",
        fontsize=12,
        y=0.98,
    )

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
