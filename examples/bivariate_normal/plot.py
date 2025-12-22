"""Plotting for bivariate Normal example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths, plot_density_contours
from .types import BivariateResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(BivariateResults, paths.load_analysis())

    sample = np.array(results["sample"])
    xs = np.array(results["plot_xs"])
    ys = np.array(results["plot_ys"])
    scipy_dens = np.array(results["scipy_densities"])
    gt_dens = np.array(results["ground_truth_densities"])
    pd_dens = np.array(results["positive_definite_densities"])
    dia_dens = np.array(results["diagonal_densities"])
    scl_dens = np.array(results["scale_densities"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Scipy vs GOAL ground truth
    plot_density_contours(
        axes[0, 0],
        xs,
        ys,
        [scipy_dens, gt_dens],
        ["Scipy", "GOAL"],
        sample,
        [colors["ground_truth"], colors["fitted"]],
    )
    axes[0, 0].set_title("Scipy vs GOAL Ground Truth")

    # Ground truth vs positive definite fit
    plot_density_contours(
        axes[0, 1],
        xs,
        ys,
        [gt_dens, pd_dens],
        ["Ground Truth", "Positive Definite"],
        sample,
    )
    axes[0, 1].set_title("Positive Definite Fit")

    # Ground truth vs diagonal fit
    plot_density_contours(
        axes[1, 0],
        xs,
        ys,
        [gt_dens, dia_dens],
        ["Ground Truth", "Diagonal"],
        sample,
    )
    axes[1, 0].set_title("Diagonal Fit")

    # Ground truth vs scale fit
    plot_density_contours(
        axes[1, 1],
        xs,
        ys,
        [gt_dens, scl_dens],
        ["Ground Truth", "Scale"],
        sample,
    )
    axes[1, 1].set_title("Scale Fit")

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
