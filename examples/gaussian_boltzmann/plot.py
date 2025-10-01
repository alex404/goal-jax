"""Plotting code for Gaussian-Boltzmann harmonium analysis."""

from typing import cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from ..shared import example_paths
from .types import GaussianBoltzmannResults

# Color palettes
POSTERIOR_COLORS = sns.color_palette("Set1", n_colors=4)
OBSERVABLE_PALETTE = sns.color_palette("twilight", as_cmap=True)


def plot_likelihood(ax: Axes, data: GaussianBoltzmannResults) -> None:
    """Plot confidence ellipses for each latent component with observations."""
    confidence_ellipses = data["component_confidence_ellipses"]
    observations = np.array(data["observations"])
    obs_x, obs_y = observations[:, 0], observations[:, 1]

    ax.scatter(obs_x, obs_y, color="black", s=10, alpha=0.5, label="Observations")

    for i, ellipse in enumerate(confidence_ellipses):
        ellipse_arr = np.array(ellipse)
        ax.plot(ellipse_arr[:, 0], ellipse_arr[:, 1], label=f"Component {i}")

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(fontsize="small")


def plot_observable_density(ax: Axes, data: GaussianBoltzmannResults) -> None:
    """Plot learned observable density with posterior observation markers."""
    plot_xs = np.array(data["plot_range_x"])
    plot_ys = np.array(data["plot_range_y"])
    density = np.array(data["learned_density"])
    posterior_obs = np.array(data["posterior_observations"])

    X1, X2 = np.meshgrid(plot_xs, plot_ys)
    heatmap = ax.contourf(X1, X2, density, levels=100, cmap=OBSERVABLE_PALETTE)

    for color, (obs1, obs2) in zip(POSTERIOR_COLORS, posterior_obs):
        ax.scatter(obs1, obs2, color=color, s=100, edgecolors="white", linewidths=2, zorder=10)

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    return heatmap


def plot_prior(ax: Axes, data: GaussianBoltzmannResults) -> None:
    """Plot prior correlation matrix from moment matrix."""
    moment_matrix = np.array(data["prior_moment_matrix"])

    # Convert moment matrix to correlation matrix
    # For binary variables: Corr(i,j) = (E[z_i z_j] - E[z_i]E[z_j]) / sqrt(Var(z_i)Var(z_j))
    # Where E[z_i] = E[z_i^2] = diagonal, and Var(z_i) = E[z_i](1-E[z_i])
    diag = np.diag(moment_matrix)
    variances = diag * (1 - diag)
    std_devs = np.sqrt(variances + 1e-8)  # Add small value to avoid division by zero

    # Compute correlation: (E[z_i z_j] - E[z_i]E[z_j]) / (std_i * std_j)
    means_outer = np.outer(diag, diag)
    correlation = (moment_matrix - means_outer) / (np.outer(std_devs, std_devs) + 1e-8)

    # Ensure diagonal is 1
    np.fill_diagonal(correlation, 1.0)

    # Plot with red-blue diverging colormap
    heatmap = ax.imshow(
        correlation,
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-1,
        vmax=1
    )
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Neuron Index")
    n = moment_matrix.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(1, n + 1))
    ax.set_yticklabels(np.arange(1, n + 1))
    ax.set_title("Prior")

    return heatmap


def plot_posterior(
    ax: gridspec.SubplotSpec,
    data: GaussianBoltzmannResults,
    fig: Figure
) -> tuple[list, Axes]:
    """Plot posterior moment matrices in 2x2 grid."""
    posterior_matrices = data["posterior_moment_matrices"]

    # Create nested GridSpec for 2x2 subplots
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=ax, wspace=0.05, hspace=0.05
    )
    heatmaps = []
    axes_list = []

    for i, matrix in enumerate(posterior_matrices):
        matrix_arr = np.array(matrix)
        inner_ax = fig.add_subplot(inner_grid[i])
        axes_list.append(inner_ax)

        heatmap = inner_ax.imshow(
            matrix_arr,
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )
        heatmaps.append(heatmap)

        inner_ax.set_xticks([])
        inner_ax.set_yticks([])

        # Add colored border matching posterior observation markers
        rect = Rectangle(
            (0, 0),
            1,
            1,
            linewidth=5,
            edgecolor=POSTERIOR_COLORS[i],
            facecolor="none",
            transform=inner_ax.transAxes,
        )
        inner_ax.add_patch(rect)

    # Add title to the subplot spec region
    # We'll use the first axis to add a suptitle-like text
    fig.text(
        0.745, 0.475, "Posterior",
        ha="center", va="top",
        fontsize=12, fontweight="normal"
    )

    return heatmaps, axes_list[0]


def plot_training_history(ax: Axes, data: GaussianBoltzmannResults) -> None:
    """Plot training log-likelihood over epochs."""
    log_likelihoods = np.array(data["log_likelihoods"])
    epochs = np.arange(len(log_likelihoods))

    ax.plot(epochs, log_likelihoods, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Log-Likelihood")
    ax.set_title("Training History")
    ax.grid(True, alpha=0.3)


def create_gaussian_boltzmann_plots(results: GaussianBoltzmannResults) -> Figure:
    """Create comprehensive visualization of Gaussian-Boltzmann harmonium results."""
    fig = plt.figure(figsize=(12, 10))

    # Create 2x2 grid matching the expected layout
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)

    # Top left: Likelihood
    ax_likelihood = fig.add_subplot(outer_grid[0, 0])

    # Top right: Prior
    ax_prior = fig.add_subplot(outer_grid[0, 1])

    # Bottom left: Observable Density
    ax_density = fig.add_subplot(outer_grid[1, 0])

    # Bottom right: Posterior (2x2 grid of heatmaps)
    ax_posterior_spec = outer_grid[1, 1]

    # Plot everything
    plot_likelihood(ax_likelihood, results)
    ax_likelihood.set_title("Likelihood")

    heatmap_prior = plot_prior(ax_prior, results)
    cbar_prior = fig.colorbar(heatmap_prior, ax=ax_prior)
    cbar_prior.set_label("Correlation")

    heatmap_density = plot_observable_density(ax_density, results)
    ax_density.set_title("Observable Density")
    cbar_density = fig.colorbar(heatmap_density, ax=ax_density)
    cbar_density.set_label("Density")

    heatmaps_posterior, ax_posterior = plot_posterior(ax_posterior_spec, results, fig)
    # Add colorbar for posterior - use one of the heatmaps
    cbar_posterior = fig.colorbar(heatmaps_posterior[0], ax=ax_posterior, fraction=0.046, pad=0.04)
    cbar_posterior.set_label("Moment")

    # Add main title
    fig.suptitle("Gaussian-Boltzmann Distribution", fontsize=16, fontweight="bold")

    return fig


def main() -> None:
    """Load results and create visualization."""
    paths = example_paths(__file__)

    # Load analysis results
    results = cast(GaussianBoltzmannResults, paths.load_analysis())

    # Create and save figure
    fig = create_gaussian_boltzmann_plots(results)
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
