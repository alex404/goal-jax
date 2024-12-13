# plot.py
"""Plotting code for LGM analysis."""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .common import LGMResults, analysis_path, results_dir


def plot_observable_density(
    ax: Axes,
    model_name: str,
    sample: NDArray[np.float64],
    true_latents: NDArray[np.float64],
    grid_points: NDArray[np.float64],
    density: NDArray[np.float64],
    bounds: tuple[float, float, float, float],
) -> None:
    """Plot observable space density with data overlay.

    Args:
        ax: Matplotlib axes for plotting
        model_name: Name of model being plotted
        sample: Data points to overlay
        true_latents: True latent variables that generated the data
        grid_points: Points where density was evaluated
        density: Density values at grid points
        bounds: Plot bounds (x_min, x_max, y_min, y_max)
    """
    x_min, x_max, y_min, y_max = bounds
    n_grid = int(np.sqrt(len(grid_points)))
    density_grid = density.reshape(n_grid, n_grid)
    xx = grid_points[:, 0].reshape(n_grid, n_grid)
    yy = grid_points[:, 1].reshape(n_grid, n_grid)

    # Plot density contours
    ax.contour(xx, yy, density_grid, levels=6, colors="r", alpha=0.5)

    # Plot samples colored by their true latent value
    scatter = ax.scatter(
        sample[:, 0],
        sample[:, 1],
        c=true_latents,
        alpha=0.5,
        s=20,
        cmap="viridis",
    )
    plt.colorbar(scatter, ax=ax, label="True Latent")

    # Plot true curve
    z = np.linspace(x_min, x_max, 100)
    ax.plot(z, z**2 / 2, "k--", alpha=0.5, label="True Curve")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{model_name} Observable Density")
    ax.set_aspect("equal")
    ax.legend()


def plot_latent_density(
    ax: Axes,
    model_name: str,
    true_latents: NDArray[np.float64],
    latent_grid: NDArray[np.float64],
    density: NDArray[np.float64],
) -> None:
    """Plot latent space density.

    Args:
        ax: Matplotlib axes for plotting
        model_name: Name of model being plotted
        true_latents: True latent variables to show as histogram
        latent_grid: Points where density was evaluated
        density: Density values at grid points
    """
    # Plot model density
    ax.plot(latent_grid, density, "r-", label="Model Density")

    # Plot histogram of true latents
    ax.hist(
        true_latents,
        bins=30,
        density=True,
        alpha=0.5,
        label="True Latents",
    )

    ax.set_title(f"{model_name} Latent Density")
    ax.set_xlabel("Latent Variable")
    ax.set_ylabel("Density")
    ax.legend()


def plot_training_histories(
    ax: Axes,
    training_lls: dict[str, list[float]],
) -> None:
    """Plot training histories for log likelihood.

    Args:
        ax: Axes for log likelihood plot
        training_lls: Log likelihood histories for each model
    """
    for model_name, lls in training_lls.items():
        steps = np.arange(len(lls))
        ax.plot(steps, lls, label=model_name)

    ax.set_xlabel("EM Step")
    ax.set_ylabel("Average Log Likelihood")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True)


def create_lgm_plots(results: LGMResults) -> Figure:
    """Create complete figure with all LGM plots.

    Args:
        results: Analysis results from LGM training

    Returns:
        Figure containing all plots
    """
    # Create figure with 2x3 subplot grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # Create density comparison axes
    ax_fa_obs = fig.add_subplot(gs[0, 0])
    ax_pca_obs = fig.add_subplot(gs[0, 1])
    ax_lat = fig.add_subplot(gs[0, 2])
    ax_ll = fig.add_subplot(gs[1, :])

    # Convert data for plotting
    sample = np.array(results["sample"])
    true_latents = np.array(results["true_latents"])
    obs_grid = np.array(results["grid_points"]["observable"])
    lat_grid = np.array(results["grid_points"]["latent"])

    # Plot observable densities
    plot_observable_density(
        ax_fa_obs,
        "Factor Analysis",
        sample,
        true_latents,
        obs_grid,
        np.array(results["observable_densities"]["factor_analysis"]),
        results["plot_bounds"],
    )
    plot_observable_density(
        ax_pca_obs,
        "PCA",
        sample,
        true_latents,
        obs_grid,
        np.array(results["observable_densities"]["pca"]),
        results["plot_bounds"],
    )

    # Plot latent densities
    plot_latent_density(
        ax_lat,
        "Factor Analysis vs PCA",
        true_latents,
        lat_grid,
        np.array(results["latent_densities"]["factor_analysis"]),
    )
    ax_lat.plot(
        lat_grid,
        np.array(results["latent_densities"]["pca"]),
        "g-",
        label="PCA Density",
    )

    # Plot training histories
    plot_training_histories(ax_ll, results["training_lls"])

    fig.tight_layout()
    return fig


def main() -> None:
    """Generate all plots from analysis results."""
    # Load results
    with open(analysis_path) as f:
        results = cast(LGMResults, json.load(f))

    # Create and save figure
    fig = create_lgm_plots(results)
    fig.savefig(results_dir / "lgm_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
