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

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Row 1: Initial densities + training
    ax_init_12 = fig.add_subplot(gs[0, 0])
    ax_init_13 = fig.add_subplot(gs[0, 1])
    ax_init_23 = fig.add_subplot(gs[0, 2])
    ax_ll = fig.add_subplot(gs[0, 3])

    # Row 2: Final densities + assignments
    ax_final_12 = fig.add_subplot(gs[1, 0])
    ax_final_13 = fig.add_subplot(gs[1, 1])
    ax_final_23 = fig.add_subplot(gs[1, 2])
    ax_assign = fig.add_subplot(gs[1, 3])

    # Initial marginals
    for ax, dim1, dim2, dens_key, title in [
        (ax_init_12, 0, 1, "initial_density_x1x2", "Initial: X1 vs X2"),
        (ax_init_13, 0, 2, "initial_density_x1x3", "Initial: X1 vs X3"),
        (ax_init_23, 1, 2, "initial_density_x2x3", "Initial: X2 vs X3"),
    ]:
        ax.scatter(samples[:, dim1], samples[:, dim2], alpha=0.3, s=10, c=gt_components, cmap="viridis")
        ax.contour(X, Y, np.array(results[dens_key]), levels=8, colors=colors["fitted"], alpha=0.6)
        ax.set_xlabel(f"X{dim1+1}")
        ax.set_ylabel(f"X{dim2+1}")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Training history
    lls = np.array(results["log_likelihoods"])
    ax_ll.plot(lls, color=colors["initial"])
    ax_ll.axhline(y=results["ground_truth_ll"], color=colors["tertiary"], linestyle="--", label="GT LL")
    ax_ll.set_xlabel("Step")
    ax_ll.set_ylabel("Log Likelihood")
    ax_ll.set_title("Training History")
    ax_ll.legend()
    ax_ll.grid(True, alpha=0.3)

    # Final marginals with ground truth overlay
    for ax, dim1, dim2, gt_key, final_key, title in [
        (ax_final_12, 0, 1, "ground_truth_density_x1x2", "final_density_x1x2", "Final: X1 vs X2"),
        (ax_final_13, 0, 2, "ground_truth_density_x1x3", "final_density_x1x3", "Final: X1 vs X3"),
        (ax_final_23, 1, 2, "ground_truth_density_x2x3", "final_density_x2x3", "Final: X2 vs X3"),
    ]:
        ax.scatter(samples[:, dim1], samples[:, dim2], alpha=0.3, s=10, c=gt_components, cmap="viridis")
        ax.contour(X, Y, np.array(results[gt_key]), levels=8, colors=colors["ground_truth"], alpha=0.5, linestyles="--")
        ax.contour(X, Y, np.array(results[final_key]), levels=8, colors=colors["fitted"], alpha=0.6)
        ax.set_xlabel(f"X{dim1+1}")
        ax.set_ylabel(f"X{dim2+1}")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Component assignments
    final_assign = np.array(results["final_assignments"])
    pred_components = np.argmax(final_assign, axis=1)
    accuracy = np.mean(pred_components == gt_components)

    sc = ax_assign.scatter(samples[:, 0], samples[:, 1], c=pred_components, cmap="viridis", alpha=0.5, s=30, edgecolors="black", linewidths=0.5)
    ax_assign.set_title(f"Predicted Components (Accuracy: {accuracy:.1%})")
    ax_assign.set_xlabel("X1")
    ax_assign.set_ylabel("X2")
    ax_assign.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax_assign, label="Component", ticks=[0, 1])

    fig.suptitle("Mixture of Factor Analyzers (3D obs, 2D latent, 2 components)", fontsize=16, fontweight="bold")
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
