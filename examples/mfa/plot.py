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

    # Define colors and line styles for each model (unified across all plots)
    style_gt = {"color": colors["ground_truth"], "linestyle": "--", "linewidth": 2}
    style_fa = {"color": colors["fitted"], "linestyle": "-", "linewidth": 2}
    style_diag = {"color": "#0096FF", "linestyle": "-.", "linewidth": 2.5}  # Bright blue

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Top-left: Training curves with ground truth reference
    ax_train = axes[0, 0]
    fa_lls = results["log_likelihoods"]["FA"]
    diag_lls = results["log_likelihoods"]["Diag"]
    gt_ll = results["ground_truth_ll"]
    n_steps = len(fa_lls)

    ax_train.axhline(gt_ll, label="Ground Truth", **style_gt)
    ax_train.plot(fa_lls, label="FA (full cov)", **style_fa)
    ax_train.plot(diag_lls, label="Diag (diag cov)", **style_diag)
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Log-Likelihood")
    ax_train.set_title("Training")
    ax_train.set_xlim(0, n_steps)
    ax_train.legend(loc="lower right", fontsize=9)
    ax_train.grid(True)

    # Remaining plots: Marginal density comparisons
    marginal_configs = [
        (axes[0, 1], 0, 1, "density_x1x2", "X1 vs X2"),
        (axes[1, 0], 0, 2, "density_x1x3", "X1 vs X3"),
        (axes[1, 1], 1, 2, "density_x2x3", "X2 vs X3"),
    ]

    # Marker styles for each component class
    markers = ["o", "s", "^"]  # circle, square, triangle

    for ax, dim1, dim2, dens_key, title in marginal_configs:
        # Scatter plot of data - black points with different markers per class
        for k in range(3):
            mask = gt_components == k
            ax.scatter(
                samples[mask, dim1],
                samples[mask, dim2],
                alpha=0.3,
                s=15,
                c="black",
                marker=markers[k],
            )

        # Ground truth contours
        ax.contour(
            X, Y,
            np.array(results[dens_key]["Ground Truth"]),
            levels=6,
            colors=style_gt["color"],
            linestyles=style_gt["linestyle"],
            linewidths=style_gt["linewidth"],
            alpha=0.8,
        )

        # FA final contours
        ax.contour(
            X, Y,
            np.array(results[dens_key]["FA Final"]),
            levels=6,
            colors=style_fa["color"],
            linestyles=style_fa["linestyle"],
            linewidths=style_fa["linewidth"],
            alpha=0.8,
        )

        # Diag final contours
        ax.contour(
            X, Y,
            np.array(results[dens_key]["Diag Final"]),
            levels=6,
            colors=style_diag["color"],
            linestyles=style_diag["linestyle"],
            linewidths=style_diag["linewidth"],
            alpha=0.8,
        )

        ax.set_xlabel(f"X{dim1 + 1}")
        ax.set_ylabel(f"X{dim2 + 1}")
        ax.set_title(title)
        ax.grid(True)

    fig.suptitle(
        "Mixture of Factor Analyzers: FA vs Diagonal Latent Covariance",
        fontsize=14,
    )
    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
