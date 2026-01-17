"""Visualization for RBM MNIST example."""

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, example_paths


def main():
    paths = example_paths(__file__)
    apply_style(paths)
    results = paths.load_analysis()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Training curves - Reconstruction Error
    ax = axes[0, 0]
    ax.plot(results["reconstruction_errors"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Error")

    # Training curves - Free Energy
    ax = axes[0, 1]
    ax.plot(results["free_energies"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Free Energy")
    ax.set_title("Mean Free Energy")

    # Weight filters (show first 64)
    ax = axes[0, 2]
    filters = np.array(results["weight_filters"])
    n_filters = min(64, len(filters))
    grid_size = int(np.ceil(np.sqrt(n_filters)))

    # Create grid of filters
    filter_grid = np.zeros((grid_size * 28, grid_size * 28))
    for i in range(n_filters):
        row = i // grid_size
        col = i % grid_size
        # Normalize filter for visualization
        f = filters[i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        filter_grid[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28] = f

    ax.imshow(filter_grid, cmap="gray")
    ax.set_title("Learned Filters")
    ax.axis("off")

    # Reconstructions
    ax = axes[1, 0]
    n_samples = min(10, len(results["original_images"]))
    comparison = np.zeros((28 * 2, 28 * n_samples))
    for i in range(n_samples):
        comparison[:28, i * 28 : (i + 1) * 28] = np.array(
            results["original_images"][i]
        ).reshape(28, 28)
        comparison[28:, i * 28 : (i + 1) * 28] = np.array(
            results["reconstructed_images"][i]
        ).reshape(28, 28)
    ax.imshow(comparison, cmap="gray")
    ax.set_title("Original (top) vs Reconstructed (bottom)")
    ax.axis("off")

    # Generated samples
    ax = axes[1, 1]
    n_gen = min(10, len(results["generated_samples"]))
    generated = np.zeros((28, 28 * n_gen))
    for i in range(n_gen):
        generated[:, i * 28 : (i + 1) * 28] = np.array(
            results["generated_samples"][i]
        ).reshape(28, 28)
    ax.imshow(generated, cmap="gray")
    ax.set_title("Generated Samples")
    ax.axis("off")

    # Hide unused subplot
    axes[1, 2].axis("off")

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
