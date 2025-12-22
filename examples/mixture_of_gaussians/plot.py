"""Plotting for mixture of Gaussians example."""

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
from .types import MixtureResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(MixtureResults, paths.load_analysis())

    sample = np.array(results["sample"])
    xs = np.array(results["plot_xs"])
    ys = np.array(results["plot_ys"])

    gt_dens = np.array(results["ground_truth_densities"])
    scipy_dens = np.array(results["scipy_densities"])
    pd_init = np.array(results["init_positive_definite_densities"])
    dia_init = np.array(results["init_diagonal_densities"])
    iso_init = np.array(results["init_isotropic_densities"])
    pd_final = np.array(results["positive_definite_densities"])
    dia_final = np.array(results["diagonal_densities"])
    iso_final = np.array(results["isotropic_densities"])

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_init = fig.add_subplot(gs[0, 1])
    ax_ll = fig.add_subplot(gs[0, 2])
    ax_iso = fig.add_subplot(gs[1, 0])
    ax_dia = fig.add_subplot(gs[1, 1])
    ax_pd = fig.add_subplot(gs[1, 2])

    # Ground truth vs sklearn
    plot_density_contours(
        ax_gt, xs, ys, [scipy_dens, gt_dens], ["Sklearn", "GOAL"], sample
    )
    ax_gt.set_title("Sklearn vs GOAL Ground Truth")

    # Initial densities
    plot_density_contours(
        ax_init,
        xs,
        ys,
        [pd_init, dia_init, iso_init],
        ["Pos. Def.", "Diagonal", "Isotropic"],
        sample,
        model_colors[:3],
    )
    ax_init.set_title("Initial Densities")

    # Training history
    plot_training_history(ax_ll, results["training_lls"])
    ax_ll.set_title("Training History")

    # Final fits
    plot_density_contours(
        ax_iso, xs, ys, [gt_dens, iso_final], ["Ground Truth", "Isotropic"], sample
    )
    ax_iso.set_title("Isotropic Fit")

    plot_density_contours(
        ax_dia, xs, ys, [gt_dens, dia_final], ["Ground Truth", "Diagonal"], sample
    )
    ax_dia.set_title("Diagonal Fit")

    plot_density_contours(
        ax_pd, xs, ys, [gt_dens, pd_final], ["Ground Truth", "Pos. Def."], sample
    )
    ax_pd.set_title("Positive Definite Fit")

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
