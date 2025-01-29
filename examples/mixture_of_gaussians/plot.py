"""Plotting code for mixture model analysis."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..shared import example_paths
from .types import MixtureResults

# built in colors
default_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
]
# black red
br_colors = ["#000000", "#FF0000"]


def plot_density_comparison(
    title: str,
    names: list[str],
    ax: Axes,
    sample: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zss: list[NDArray[np.float64]],
    colors: list[str] = default_colors,
) -> None:
    # Find density plot ranges
    min_val = min(xs.min(), ys.min())
    max_val = max(xs.max(), ys.max())
    # Default colors

    # Plot samples
    ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=10, label="Samples")

    # Plot contours for both distributions
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


def create_mixture_plots(results: MixtureResults) -> Figure:
    # Create figure with 3x2 subplot grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # Create density comparison axes
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_ini = fig.add_subplot(gs[0, 1])
    ax_ll = fig.add_subplot(gs[0, 2])
    ax_iso = fig.add_subplot(gs[1, 0])
    ax_diag = fig.add_subplot(gs[1, 1])
    ax_pd = fig.add_subplot(gs[1, 2])

    # Create training history axes

    # Convert data for plotting
    sample = np.array(results["sample"])
    xs = np.array(results["plot_xs"])
    ys = np.array(results["plot_ys"])
    gt_dens = np.array(results["ground_truth_densities"])
    pod_dens0 = np.array(results["init_positive_definite_densities"])
    dia_dens0 = np.array(results["init_diagonal_densities"])
    iso_dens0 = np.array(results["init_isotropic_densities"])
    pod_dens1 = np.array(results["positive_definite_densities"])
    dia_dens1 = np.array(results["diagonal_densities"])
    iso_dens1 = np.array(results["isotropic_densities"])
    scipy_dens = np.array(results["scipy_densities"])

    # Plot density comparisons
    plot_density_comparison(
        "Scipy vs. Goal Ground Truth",
        ["Scipy", "Goal"],
        ax_gt,
        sample,
        xs,
        ys,
        [scipy_dens, gt_dens],
        colors=br_colors,
    )
    plot_density_comparison(
        "Positive Definite Fit",
        ["Ground Truth", "Positive Definite"],
        ax_pd,
        sample,
        xs,
        ys,
        [gt_dens, pod_dens1],
        colors=br_colors,
    )
    plot_density_comparison(
        "Diagonal Fit",
        ["Ground Truth", "Diagonal"],
        ax_diag,
        sample,
        xs,
        ys,
        [gt_dens, dia_dens1],
        colors=br_colors,
    )
    plot_density_comparison(
        "Isotropic Fit",
        ["Ground Truth", "Isotropic"],
        ax_iso,
        sample,
        xs,
        ys,
        [gt_dens, iso_dens1],
        colors=br_colors,
    )

    plot_density_comparison(
        "Initial Densities",
        ["Positive Definite", "Diagonal", "Isotropic"],
        ax_ini,
        sample,
        xs,
        ys,
        [pod_dens0, dia_dens0, iso_dens0],
    )

    # Plot training histories
    plot_training_histories(ax_ll, results["training_lls"])

    fig.tight_layout()
    return fig


def main() -> None:
    paths = example_paths(__file__)
    # Load results
    analysis = cast(MixtureResults, paths.load_analysis())

    # Create and save figure
    fig = create_mixture_plots(analysis)
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
