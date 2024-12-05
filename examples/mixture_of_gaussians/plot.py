"""Plotting code for mixture model analysis."""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .common import MixtureResults, analysis_path, results_dir


def plot_density_comparison(
    target_name: str,
    model_name: str,
    ax: Axes,
    sample: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    target_zs: NDArray[np.float64],
    model_zs: NDArray[np.float64],
) -> None:
    """Plot comparison between target and model densities with data overlay.

    Args:
        target_name: Name of target distribution
        model_name: Name of model being compared
        ax: Matplotlib axes for plotting
        sample: Data points to overlay
        xs, ys: Grid coordinates for density evaluation
        target_zs: Target density values on grid
        model_zs: Model density values on grid
    """
    # Find density plot ranges
    min_val = min(xs.min(), ys.min())
    max_val = max(xs.max(), ys.max())

    # Plot samples
    ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=10, label="Samples")

    # Plot contours for both distributions
    ax.contour(xs, ys, target_zs, levels=6, colors="b", alpha=0.5)
    ax.plot([], [], color="b", label=target_name)

    ax.contour(xs, ys, model_zs, levels=6, colors="r", alpha=0.5)
    ax.plot([], [], color="r", label=model_name)

    # Configure plot
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(f"{model_name} vs {target_name}")


def plot_training_histories(
    ax_ll: Axes,
    training_lls: dict[str, list[float]],
) -> None:
    """Plot training histories for log likelihood and relative entropy.

    Args:
        ax_ll: Axes for log likelihood plot
        ax_re: Axes for relative entropy plot
        training_lls: Log likelihood histories for each model
    """
    # Plot histories for each model
    for model_name, lls in training_lls.items():
        steps = np.arange(len(lls))
        ax_ll.plot(steps, lls, label=model_name)

    # Configure log likelihood plot
    ax_ll.set_xlabel("EM Step")
    ax_ll.set_ylabel("Average Log Likelihood")
    ax_ll.set_title("Training History")
    ax_ll.legend()
    ax_ll.grid(True)


def create_mixture_plots(results: MixtureResults) -> Figure:
    """Create complete figure with all mixture model plots.

    Args:
        results: Analysis results from mixture model training

    Returns:
        Figure containing all plots
    """
    # Create figure with 3x2 subplot grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # Create density comparison axes
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pd = fig.add_subplot(gs[0, 1])
    ax_diag = fig.add_subplot(gs[0, 2])
    ax_iso = fig.add_subplot(gs[1, 0])

    # Create training history axes
    ax_ll = fig.add_subplot(gs[1, 1:])

    # Convert data for plotting
    sample = np.array(results["sample"])
    xs = np.array(results["plot_xs"])
    ys = np.array(results["plot_ys"])
    gt_dens = np.array(results["ground_truth_densities"])
    pd_dens = np.array(results["positive_definite_densities"])
    diag_dens = np.array(results["diagonal_densities"])
    iso_dens = np.array(results["isotropic_densities"])
    scipy_dens = np.array(results["scipy_densities"])

    # Plot density comparisons
    plot_density_comparison(
        "Scipy Ground Truth",
        "Goal Ground Truth",
        ax_gt,
        sample,
        xs,
        ys,
        scipy_dens,
        gt_dens,
    )
    plot_density_comparison(
        "Ground Truth", "Positive Definite", ax_pd, sample, xs, ys, gt_dens, pd_dens
    )
    plot_density_comparison(
        "Ground Truth", "Diagonal", ax_diag, sample, xs, ys, gt_dens, diag_dens
    )
    plot_density_comparison(
        "Ground Truth", "Isotropic", ax_iso, sample, xs, ys, gt_dens, iso_dens
    )

    # Plot training histories
    plot_training_histories(ax_ll, results["training_lls"])

    fig.tight_layout()
    return fig


def main() -> None:
    """Generate all plots from analysis results."""
    # Load results
    with open(analysis_path) as f:
        results = cast(MixtureResults, json.load(f))

    # Create and save figure
    fig = create_mixture_plots(results)
    fig.savefig(results_dir / "mixture_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
