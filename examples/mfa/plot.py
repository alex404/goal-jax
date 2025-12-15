"""Plotting for Mixture of Factor Analyzers example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..shared import example_paths
from .types import MFAResults


def create_mfa_plots(results: MFAResults) -> Figure:
    """Create comprehensive MFA visualization.

    Layout: 2 rows x 4 columns
    - Row 1: Initial densities (3 marginals) + training history
    - Row 2: Final densities (3 marginals) + component assignments
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Convert to numpy
    samples = np.array(results["observations"])
    gt_components = np.array(results["ground_truth_components"])
    x_range = np.array(results["plot_range"])
    X, Y = np.meshgrid(x_range, x_range)

    # Row 1: Initial fitted densities
    ax_init_12 = fig.add_subplot(gs[0, 0])
    ax_init_13 = fig.add_subplot(gs[0, 1])
    ax_init_23 = fig.add_subplot(gs[0, 2])
    ax_ll = fig.add_subplot(gs[0, 3])

    # Row 2: Final fitted densities + component assignments
    ax_final_12 = fig.add_subplot(gs[1, 0])
    ax_final_13 = fig.add_subplot(gs[1, 1])
    ax_final_23 = fig.add_subplot(gs[1, 2])
    ax_resp = fig.add_subplot(gs[1, 3])

    # Plot initial marginals with samples
    init_dens_12 = np.array(results["initial_density_x1x2"])
    init_dens_13 = np.array(results["initial_density_x1x3"])
    init_dens_23 = np.array(results["initial_density_x2x3"])

    # X1-X2
    ax_init_12.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c=gt_components, cmap="viridis")
    ax_init_12.contour(X, Y, init_dens_12, levels=8, colors="red", alpha=0.6)
    ax_init_12.set_title("Initial: X1 vs X2 (X3=0)")
    ax_init_12.set_xlabel("X1")
    ax_init_12.set_ylabel("X2")
    ax_init_12.grid(True, alpha=0.3)

    # X1-X3
    ax_init_13.scatter(samples[:, 0], samples[:, 2], alpha=0.3, s=10, c=gt_components, cmap="viridis")
    ax_init_13.contour(X, Y, init_dens_13, levels=8, colors="red", alpha=0.6)
    ax_init_13.set_title("Initial: X1 vs X3 (X2=0)")
    ax_init_13.set_xlabel("X1")
    ax_init_13.set_ylabel("X3")
    ax_init_13.grid(True, alpha=0.3)

    # X2-X3
    ax_init_23.scatter(samples[:, 1], samples[:, 2], alpha=0.3, s=10, c=gt_components, cmap="viridis")
    ax_init_23.contour(X, Y, init_dens_23, levels=8, colors="red", alpha=0.6)
    ax_init_23.set_title("Initial: X2 vs X3 (X1=0)")
    ax_init_23.set_xlabel("X2")
    ax_init_23.set_ylabel("X3")
    ax_init_23.grid(True, alpha=0.3)

    # Plot training history
    lls = np.array(results["log_likelihoods"])
    ax_ll.plot(lls, linewidth=2)
    ax_ll.set_xlabel("EM Iteration")
    ax_ll.set_ylabel("Avg Log Likelihood")
    ax_ll.set_title("Training History")
    ax_ll.grid(True, alpha=0.3)
    ax_ll.axhline(y=results["ground_truth_ll"], color="green", linestyle="--", label="Final LL")
    ax_ll.legend()

    # Plot final marginals with samples AND ground truth densities
    final_dens_12 = np.array(results["final_density_x1x2"])
    final_dens_13 = np.array(results["final_density_x1x3"])
    final_dens_23 = np.array(results["final_density_x2x3"])

    gt_dens_12 = np.array(results["ground_truth_density_x1x2"])
    gt_dens_13 = np.array(results["ground_truth_density_x1x3"])
    gt_dens_23 = np.array(results["ground_truth_density_x2x3"])

    # X1-X2
    ax_final_12.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c=gt_components, cmap="viridis")
    ax_final_12.contour(X, Y, gt_dens_12, levels=8, colors="green", alpha=0.5, linestyles="--", linewidths=2)
    ax_final_12.contour(X, Y, final_dens_12, levels=8, colors="blue", alpha=0.6)
    ax_final_12.set_title("Final vs GT: X1 vs X2 (X3=0)")
    ax_final_12.set_xlabel("X1")
    ax_final_12.set_ylabel("X2")
    ax_final_12.grid(True, alpha=0.3)
    ax_final_12.legend(
        [plt.Line2D([0], [0], color="green", linestyle="--", linewidth=2),
         plt.Line2D([0], [0], color="blue", linewidth=1)],
        ["Ground Truth", "Fitted"],
        loc="upper right",
        fontsize=8
    )

    # X1-X3
    ax_final_13.scatter(samples[:, 0], samples[:, 2], alpha=0.3, s=10, c=gt_components, cmap="viridis")
    ax_final_13.contour(X, Y, gt_dens_13, levels=8, colors="green", alpha=0.5, linestyles="--", linewidths=2)
    ax_final_13.contour(X, Y, final_dens_13, levels=8, colors="blue", alpha=0.6)
    ax_final_13.set_title("Final vs GT: X1 vs X3 (X2=0)")
    ax_final_13.set_xlabel("X1")
    ax_final_13.set_ylabel("X3")
    ax_final_13.grid(True, alpha=0.3)

    # X2-X3
    ax_final_23.scatter(samples[:, 1], samples[:, 2], alpha=0.3, s=10, c=gt_components, cmap="viridis")
    ax_final_23.contour(X, Y, gt_dens_23, levels=8, colors="green", alpha=0.5, linestyles="--", linewidths=2)
    ax_final_23.contour(X, Y, final_dens_23, levels=8, colors="blue", alpha=0.6)
    ax_final_23.set_title("Final vs GT: X2 vs X3 (X1=0)")
    ax_final_23.set_xlabel("X2")
    ax_final_23.set_ylabel("X3")
    ax_final_23.grid(True, alpha=0.3)

    # Plot component assignments (predicted vs ground truth)
    final_resp = np.array(results["final_responsibilities"])
    pred_components = np.argmax(final_resp, axis=1)

    # Compute accuracy
    accuracy = np.mean(pred_components == gt_components)

    # Scatter plot colored by predicted component
    ax_resp.scatter(
        samples[:, 0],
        samples[:, 1],
        c=pred_components,
        cmap="viridis",
        alpha=0.5,
        s=30,
        edgecolors="black",
        linewidths=0.5,
    )
    ax_resp.set_title(f"Predicted Components (Accuracy: {accuracy:.1%})")
    ax_resp.set_xlabel("X1")
    ax_resp.set_ylabel("X2")
    ax_resp.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(
        ax_resp.collections[0],
        ax=ax_resp,
        label="Component",
        ticks=[0, 1],
    )

    fig.suptitle(
        "Mixture of Factor Analyzers (3D obs, 2D latent, 2 components)",
        fontsize=16,
        fontweight="bold",
    )

    return fig


def main():
    """Main entry point."""
    paths = example_paths(__file__)
    results = cast(MFAResults, paths.load_analysis())
    fig = create_mfa_plots(results)
    paths.save_plot(fig)
    print(f"Plot saved to {paths.results_dir / 'plot.png'}")


if __name__ == "__main__":
    main()
