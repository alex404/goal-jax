"""Plotting for dimensionality reduction comparison."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import (
    apply_style,
    example_paths,
    model_colors,
    plot_density_contours,
    plot_training_history,
)
from .types import LGMResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(LGMResults, paths.load_analysis())

    sample = np.array(results["sample"])
    grid = np.array(results["grid_points"]["observable"])
    n = int(np.sqrt(len(grid)))
    xs = grid[:, 0].reshape(n, n)
    ys = grid[:, 1].reshape(n, n)

    gt_dens = np.array(results["ground_truth_densities"]).reshape(n, n)
    fa_init = np.array(results["initial_densities"]["factor_analysis"]).reshape(n, n)
    pca_init = np.array(results["initial_densities"]["pca"]).reshape(n, n)
    fa_final = np.array(results["learned_densities"]["factor_analysis"]).reshape(n, n)
    pca_final = np.array(results["learned_densities"]["pca"]).reshape(n, n)

    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 2)

    ax_init = fig.add_subplot(gs[0, 0])
    ax_ll = fig.add_subplot(gs[0, 1])
    ax_pca = fig.add_subplot(gs[1, 0])
    ax_fa = fig.add_subplot(gs[1, 1])

    # Initial densities
    plot_density_contours(
        ax_init,
        xs,
        ys,
        [fa_init, pca_init],
        ["FA", "PCA"],
        sample,
        model_colors[:2],
    )
    ax_init.set_title("Initial Densities")

    # Training history
    plot_training_history(ax_ll, results["training_lls"])
    ax_ll.set_title("Training History")

    # Final comparisons
    plot_density_contours(
        ax_pca, xs, ys, [gt_dens, pca_final], ["Ground Truth", "PCA"], sample
    )
    ax_pca.set_title("PCA vs Ground Truth")

    plot_density_contours(
        ax_fa, xs, ys, [gt_dens, fa_final], ["Ground Truth", "FA"], sample
    )
    ax_fa.set_title("Factor Analysis vs Ground Truth")

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
