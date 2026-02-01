"""Plotting for Mixture of Factor Analyzers."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths
from .types import MFAResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(MFAResults, paths.load_analysis())

    samples = np.array(results["observations"])
    gt_components = np.array(results["ground_truth_components"])
    x_range = np.array(results["plot_range"])
    X, Y = np.meshgrid(x_range, x_range)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    # Row 1: Initial densities
    for ax, dim1, dim2, dens_key, title in [
        (axes[0, 0], 0, 1, "initial_density_x1x2", "Initial: X1 vs X2"),
        (axes[0, 1], 0, 2, "initial_density_x1x3", "Initial: X1 vs X3"),
        (axes[0, 2], 1, 2, "initial_density_x2x3", "Initial: X2 vs X3"),
    ]:
        ax.scatter(
            samples[:, dim1],
            samples[:, dim2],
            alpha=0.3,
            s=10,
            c=gt_components,
            cmap="viridis",
        )
        ax.contour(
            X,
            Y,
            np.array(results[dens_key]),
            levels=8,
            colors=colors["fitted"],
            alpha=0.6,
        )
        ax.set_xlabel(f"X{dim1 + 1}")
        ax.set_ylabel(f"X{dim2 + 1}")
        ax.set_title(title)
        ax.grid(True)

    # Row 2: Final densities with ground truth overlay
    for ax, dim1, dim2, gt_key, final_key, title in [
        (axes[1, 0], 0, 1, "ground_truth_density_x1x2", "final_density_x1x2", "Final: X1 vs X2"),
        (axes[1, 1], 0, 2, "ground_truth_density_x1x3", "final_density_x1x3", "Final: X1 vs X3"),
        (axes[1, 2], 1, 2, "ground_truth_density_x2x3", "final_density_x2x3", "Final: X2 vs X3"),
    ]:
        ax.scatter(
            samples[:, dim1],
            samples[:, dim2],
            alpha=0.3,
            s=10,
            c=gt_components,
            cmap="viridis",
        )
        ax.contour(
            X,
            Y,
            np.array(results[gt_key]),
            levels=8,
            colors=colors["ground_truth"],
            alpha=0.5,
            linestyles="--",
        )
        ax.contour(
            X,
            Y,
            np.array(results[final_key]),
            levels=8,
            colors=colors["fitted"],
            alpha=0.6,
        )
        ax.set_xlabel(f"X{dim1 + 1}")
        ax.set_ylabel(f"X{dim2 + 1}")
        ax.set_title(title)
        ax.grid(True)

    fig.suptitle(
        "Mixture of Factor Analyzers (3D obs, 2D latent, 3 components)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
