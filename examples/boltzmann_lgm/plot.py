"""Plotting for Gaussian-Boltzmann harmonium."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from ..shared import apply_style, example_paths, model_colors
from .types import GaussianBoltzmannResults


def compute_correlation(moment_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert moment matrix to correlation matrix."""
    diag = np.diag(moment_matrix)
    variances = diag * (1 - diag)
    std_devs = np.sqrt(variances + 1e-8)
    means_outer = np.outer(diag, diag)
    correlation = (moment_matrix - means_outer) / (np.outer(std_devs, std_devs) + 1e-8)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(GaussianBoltzmannResults, paths.load_analysis())

    fig = plt.figure(figsize=(9, 7.5))
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)

    # Likelihood (confidence ellipses)
    ax_lkl = fig.add_subplot(outer_grid[0, 0])
    obs = np.array(results["observations"])
    ax_lkl.scatter(obs[:, 0], obs[:, 1], color="black", s=10, alpha=0.5, label="Observations")
    for i, ellipse in enumerate(results["component_confidence_ellipses"]):
        arr = np.array(ellipse)
        ax_lkl.plot(arr[:, 0], arr[:, 1], label=f"Component {i}")
    ax_lkl.set_xlabel("x₁")
    ax_lkl.set_ylabel("x₂")
    ax_lkl.set_title("Likelihood")
    ax_lkl.legend(fontsize="small", ncol=2)

    # Prior correlation
    ax_prior = fig.add_subplot(outer_grid[0, 1])
    prior_moment = np.array(results["prior_moment_matrix"])
    prior_corr = compute_correlation(prior_moment)
    im_prior = ax_prior.imshow(prior_corr, cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    ax_prior.set_xlabel("Neuron Index")
    ax_prior.set_ylabel("Neuron Index")
    ax_prior.set_title("Prior Correlation")
    n = prior_moment.shape[0]
    ax_prior.set_xticks(np.arange(n))
    ax_prior.set_yticks(np.arange(n))
    ax_prior.set_xticklabels(np.arange(1, n + 1))
    ax_prior.set_yticklabels(np.arange(1, n + 1))
    plt.colorbar(im_prior, ax=ax_prior, label="Correlation")

    # Observable density
    ax_dens = fig.add_subplot(outer_grid[1, 0])
    plot_xs = np.array(results["plot_range_x"])
    plot_ys = np.array(results["plot_range_y"])
    density = np.array(results["learned_density"])
    posterior_obs = np.array(results["posterior_observations"])
    xx, yy = np.meshgrid(plot_xs, plot_ys)
    heatmap = ax_dens.contourf(xx, yy, density, levels=100, cmap="viridis")
    for i, (obs_x, obs_y) in enumerate(posterior_obs):
        ax_dens.scatter(obs_x, obs_y, color=model_colors[i], s=100, edgecolors="white", linewidths=2, zorder=10)
    ax_dens.set_xlabel("x₁")
    ax_dens.set_ylabel("x₂")
    ax_dens.set_title("Observable Density")
    plt.colorbar(heatmap, ax=ax_dens, label="Density")

    # Posterior moments (2x2 grid)
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[1, 1], wspace=0.05, hspace=0.05)
    posterior_matrices = results["posterior_moment_matrices"]
    posterior_axes = []
    im = None
    for i, matrix in enumerate(posterior_matrices):
        inner_ax = fig.add_subplot(inner_grid[i])
        posterior_axes.append(inner_ax)
        im = inner_ax.imshow(np.array(matrix), cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        inner_ax.set_xticks([])
        inner_ax.set_yticks([])
        rect = Rectangle((0, 0), 1, 1, linewidth=5, edgecolor=model_colors[i], facecolor="none", transform=inner_ax.transAxes)
        inner_ax.add_patch(rect)

    fig.text(0.745, 0.475, "Posterior Moments", ha="center", va="top", fontsize=12)
    if im is not None:
        plt.colorbar(im, ax=posterior_axes, fraction=0.046, pad=0.04, label="Moment")

    fig.suptitle("Gaussian-Boltzmann Distribution", fontsize=16, fontweight="bold")
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
