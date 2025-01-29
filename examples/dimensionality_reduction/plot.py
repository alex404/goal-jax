# plot.py
"""Plotting code for LGM analysis."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..shared import example_paths
from .types import LGMResults

# Default colors for consistent styling
br_colors = ["#000000", "#FF0000"]  # black, red
default_colors = ["tab:blue", "tab:orange"]  # For initial densities


def plot_density_comparison(
    title: str,
    names: list[str],
    ax: Axes,
    sample: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zss: list[NDArray[np.float64]],
    colors: list[str] = br_colors,
) -> None:
    """Plot comparison between model densities with data overlay.

    Args:
        title: Plot title
        names: Names of densities being compared
        ax: Matplotlib axes for plotting
        sample: Data points to overlay
        xs, ys: Grid coordinates for density evaluation
        zss: List of density values on grid for each model
        colors: Colors to use for each density
    """
    # Find density plot ranges
    min_val = min(xs.min(), ys.min())
    max_val = max(xs.max(), ys.max())

    # Plot samples
    ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=10, label="Samples")

    # Plot contours for each density
    for name, zs, color in zip(names, zss, colors):
        ax.contour(xs, ys, zs, levels=6, colors=color, alpha=0.5)
        ax.plot([], [], color=color, label=name)

    # Configure plot
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_training_histories(
    ax_ll: Axes,
    training_lls: dict[str, list[float]],
) -> None:
    """Plot training histories for log likelihood.

    Args:
        ax_ll: Axes for log likelihood plot
        training_lls: Log likelihood histories for each model
    """
    # Plot histories for each model
    for (model_name, lls), color in zip(training_lls.items(), default_colors):
        steps = np.arange(len(lls))
        ax_ll.plot(steps, lls, label=model_name, color=color)

    # Configure log likelihood plot
    ax_ll.set_xlabel("EM Step")
    ax_ll.set_ylabel("Average Log Likelihood")
    ax_ll.set_title("Training History")
    ax_ll.legend()
    ax_ll.grid(True)


def create_lgm_plots(results: LGMResults) -> Figure:
    """Create complete figure with all LGM plots.

    Args:
        results: Analysis results from LGM training

    Returns:
        Figure containing all plots
    """
    # Create figure with 2x2 subplot grid
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)

    # Create axes
    ax_init = fig.add_subplot(gs[0, 0])  # Initial densities
    ax_ll = fig.add_subplot(gs[0, 1])  # Training histories
    ax_pca = fig.add_subplot(gs[1, 0])  # PCA vs ground truth
    ax_fa = fig.add_subplot(gs[1, 1])  # FA vs ground truth

    # Convert data for plotting
    sample = np.array(results["sample"])
    grid_points = np.array(results["grid_points"]["observable"])
    n_grid = int(np.sqrt(len(grid_points)))
    xs = grid_points[:, 0].reshape(n_grid, n_grid)
    ys = grid_points[:, 1].reshape(n_grid, n_grid)

    # Get initial densities
    gt_dens = np.array(results["ground_truth_densities"]).reshape(n_grid, n_grid)
    fa_dens0 = np.array(results["initial_densities"]["factor_analysis"]).reshape(
        n_grid, n_grid
    )
    pca_dens0 = np.array(results["initial_densities"]["pca"]).reshape(n_grid, n_grid)

    # Get learned densities
    fa_dens1 = np.array(results["learned_densities"]["factor_analysis"]).reshape(
        n_grid, n_grid
    )
    pca_dens1 = np.array(results["learned_densities"]["pca"]).reshape(n_grid, n_grid)

    # Plot comparisons
    plot_density_comparison(
        "Initial Densities",
        ["Factor Analysis", "PCA"],
        ax_init,
        sample,
        xs,
        ys,
        [fa_dens0, pca_dens0],
        colors=default_colors,
    )

    plot_density_comparison(
        "PCA vs Ground Truth",
        ["Initial", "Learned"],
        ax_pca,
        sample,
        xs,
        ys,
        [pca_dens1, gt_dens],
    )

    plot_density_comparison(
        "Factor Analysis vs Ground Truth",
        ["Initial", "Learned"],
        ax_fa,
        sample,
        xs,
        ys,
        [fa_dens1, gt_dens],
    )

    # Plot training histories
    plot_training_histories(ax_ll, results["training_lls"])

    fig.tight_layout()
    return fig


def main() -> None:
    """Generate all plots from analysis results."""
    # Load results
    paths = example_paths(__file__)
    analysis = cast(LGMResults, paths.load_analysis())

    # Create and save figure
    fig = create_lgm_plots(analysis)
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
