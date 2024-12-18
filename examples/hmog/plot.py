"""Plotting code for hierarchical mixture of Gaussians analysis."""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..shared import initialize_paths
from .types import HMoGResults

# black red for comparing true vs model
br_colors = ["#000000", "#FF0000"]


def plot_density_comparison(
    title: str,
    names: list[str],
    ax: Axes,
    sample: NDArray[np.float64],
    x1s: NDArray[np.float64],
    x2s: NDArray[np.float64],
    densities: list[NDArray[np.float64]],
    colors: list[str] = br_colors,
    plot_samples: bool = True,
) -> None:
    """Plot 2D density contours with optional sample points."""
    # Find density plot ranges
    min_val = min(x1s.min(), x2s.min())
    max_val = max(x1s.max(), x2s.max())

    # Plot samples if requested
    if plot_samples:
        ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=10, label="Samples")

    # Plot contours for distributions
    for name, density, color in zip(names, densities, colors):
        ax.contour(x1s, x2s, density, levels=6, colors=color, alpha=0.5)
        ax.plot([], [], color=color, label=name)

    # Configure plot
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_latent_density(
    title: str,
    names: list[str],
    ax: Axes,
    ys: NDArray[np.float64],
    densities: list[NDArray[np.float64]],
    colors: list[str] = br_colors,
) -> None:
    """Plot 1D latent density curves."""
    for name, density, color in zip(names, densities, colors):
        ax.plot(ys, density, color=color, label=name)

    ax.set_xlabel("y")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_training_history(
    ax: Axes,
    lls: NDArray[np.float64],
) -> None:
    """Plot training log likelihood history."""
    steps = np.arange(len(lls))
    ax.plot(steps, lls, color="black")

    ax.set_xlabel("EM Step")
    ax.set_ylabel("Average Log Likelihood")
    ax.set_title("Training History")
    ax.grid(True)


def create_hmog_plots(results: HMoGResults) -> Figure:
    """Create figure showing HMoG analysis results with multiple models."""
    # Create figure with 2x4 subplot grid
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)

    # Create axes for each panel
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_iso_init = fig.add_subplot(gs[0, 1])
    ax_dia_init = fig.add_subplot(gs[0, 2])
    ax_ll = fig.add_subplot(gs[0, 3])
    ax_iso_final = fig.add_subplot(gs[1, 0])
    ax_dia_final = fig.add_subplot(gs[1, 1])
    ax_lat_init = fig.add_subplot(gs[1, 2])
    ax_lat_final = fig.add_subplot(gs[1, 3])

    # Convert data for plotting
    sample = np.array(results["observations"])
    x1s = np.array(results["plot_range_x1"])
    x2s = np.array(results["plot_range_x2"])
    ys = np.array(results["plot_range_y"])
    x1_mesh, x2_mesh = np.meshgrid(x1s, x2s)

    # Get densities
    true_dens = np.array(results["observable_densities"]["Ground Truth"])
    iso_init_dens = np.array(results["observable_densities"]["Isotropic Initial"])
    iso_final_dens = np.array(results["observable_densities"]["Isotropic Final"])
    dia_init_dens = np.array(results["observable_densities"]["Diagonal Initial"])
    dia_final_dens = np.array(results["observable_densities"]["Diagonal Final"])

    # Get mixture densities
    true_mix = np.array(results["mixture_densities"]["Ground Truth"])
    iso_init_mix = np.array(results["mixture_densities"]["Isotropic Initial"])
    iso_final_mix = np.array(results["mixture_densities"]["Isotropic Final"])
    dia_init_mix = np.array(results["mixture_densities"]["Diagonal Initial"])
    dia_final_mix = np.array(results["mixture_densities"]["Diagonal Final"])

    # Get log likelihoods
    iso_lls = np.array(results["log_likelihoods"]["Isotropic"])
    dia_lls = np.array(results["log_likelihoods"]["Diagonal"])

    # Plot initial observable distributions
    plot_density_comparison(
        "Ground Truth",
        ["True"],
        ax_gt,
        sample,
        x1_mesh,
        x2_mesh,
        [true_dens],
    )

    # Plot initial observable distributions
    plot_density_comparison(
        "Initial Isotropic Model",
        ["True", "Initial"],
        ax_iso_init,
        sample,
        x1_mesh,
        x2_mesh,
        [true_dens, iso_init_dens],
    )

    plot_density_comparison(
        "Initial Diagonal Model",
        ["True", "Initial"],
        ax_dia_init,
        sample,
        x1_mesh,
        x2_mesh,
        [true_dens, dia_init_dens],
    )

    # Plot final observable distributions
    plot_density_comparison(
        "Final Isotropic Model",
        ["True", "Final"],
        ax_iso_final,
        sample,
        x1_mesh,
        x2_mesh,
        [true_dens, iso_final_dens],
    )

    plot_density_comparison(
        "Final Diagonal Model",
        ["True", "Final"],
        ax_dia_final,
        sample,
        x1_mesh,
        x2_mesh,
        [true_dens, dia_final_dens],
    )

    # Plot training histories
    steps = np.arange(len(iso_lls))
    ax_ll.plot(steps, iso_lls, label="Isotropic", color="tab:blue")
    ax_ll.plot(steps, dia_lls, label="Diagonal", color="tab:orange")
    ax_ll.set_xlabel("EM Step")
    ax_ll.set_ylabel("Average Log Likelihood")
    ax_ll.set_title("Training History")
    ax_ll.grid(True)
    ax_ll.legend()

    # Plot initial latent distributions
    plot_latent_density(
        "Initial Latent Distributions",
        ["True", "Isotropic", "Diagonal"],
        ax_lat_init,
        ys,
        [true_mix, iso_init_mix, dia_init_mix],
        colors=["black", "tab:blue", "tab:orange"],
    )

    # Plot final latent distributions
    plot_latent_density(
        "Final Latent Distributions",
        ["True", "Isotropic", "Diagonal"],
        ax_lat_final,
        ys,
        [true_mix, iso_final_mix, dia_final_mix],
        colors=["black", "tab:blue", "tab:orange"],
    )

    fig.tight_layout()
    return fig


def main() -> None:
    """Load results and create plots."""
    # Load results
    paths = initialize_paths(__file__)
    with open(paths.analysis_path) as f:
        results = cast(HMoGResults, json.load(f))

    # Create and save figure
    fig = create_hmog_plots(results)
    fig.savefig(paths.results_dir / "hmog_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
