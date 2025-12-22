"""Plotting for COM-Poisson mixture analysis."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..shared import apply_style, example_paths, model_colors, plot_training_history
from .types import PoissonAnalysisResults


def plot_scatter_comparison(
    ax: Axes,
    sample_vals: NDArray[np.float64],
    model_vals: NDArray[np.float64],
    color: str,
    label: str,
) -> None:
    """Plot scatter comparison with diagonal line."""
    data_min = min(float(sample_vals.min()), float(model_vals.min()))
    data_max = max(float(sample_vals.max()), float(model_vals.max()))
    padding = (data_max - data_min) * 0.05
    data_range = (data_min - padding, data_max + padding)

    ax.scatter(model_vals, sample_vals, s=30, color=color, label=label, alpha=0.7)
    ax.plot(data_range, data_range, "k--", alpha=0.5)
    ax.set_xlim(data_range)
    ax.set_ylim(data_range)
    ax.set_aspect("equal", "box")


def plot_correlation_matrix(
    ax: Axes,
    sample_corr: NDArray[np.float64],
    model_corr: NDArray[np.float64],
    title: str,
) -> plt.cm.ScalarMappable:
    """Plot split correlation matrix (sample lower, model upper)."""
    n = sample_corr.shape[0]
    combined = np.zeros_like(sample_corr)

    for i in range(n):
        for j in range(i + 1):
            combined[i, j] = sample_corr[i, j]
    for i in range(n):
        for j in range(i + 1, n):
            combined[i, j] = model_corr[i, j]

    im = ax.imshow(combined, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    return im


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(PoissonAnalysisResults, paths.load_analysis())

    sample_corr = np.array(results["sample_stats"]["correlation_matrix"])
    fa_corr = np.array(results["fan_stats"]["correlation_matrix"])
    psn_corr = np.array(results["psn_stats"]["correlation_matrix"])
    com_corr = np.array(results["com_stats"]["correlation_matrix"])

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    ax_means = fig.add_subplot(gs[0, 0])
    ax_covs = fig.add_subplot(gs[0, 1])
    ax_ll = fig.add_subplot(gs[0, 2])
    ax_fa = fig.add_subplot(gs[1, 0])
    ax_psn = fig.add_subplot(gs[1, 1])
    ax_com = fig.add_subplot(gs[1, 2])

    # Means comparison
    models = [
        ("FA", results["fan_stats"], model_colors[0]),
        ("Poisson", results["psn_stats"], model_colors[1]),
        ("COM-Poisson", results["com_stats"], model_colors[2]),
    ]
    sample_means = np.array(results["sample_stats"]["means"])
    for name, stats, color in models:
        plot_scatter_comparison(ax_means, sample_means, np.array(stats["means"]), color, name)
    ax_means.set_xlabel("Model Means")
    ax_means.set_ylabel("Sample Means")
    ax_means.legend()

    # Covariances comparison
    sample_covs = np.array(results["sample_stats"]["covariances"])
    for name, stats, color in models:
        plot_scatter_comparison(ax_covs, sample_covs, np.array(stats["covariances"]), color, name)
    ax_covs.set_xlabel("Model Covariances")
    ax_covs.set_ylabel("Sample Covariances")
    ax_covs.legend()

    # Training history
    plot_training_history(
        ax_ll,
        {
            "Factor Analysis": results["fan_lls"],
            "Poisson Mixture": results["psn_lls"],
            "COM-Poisson": results["com_lls"],
        },
    )
    ax_ll.set_title("Training History")

    # Correlation matrices
    im_fa = plot_correlation_matrix(ax_fa, sample_corr, fa_corr, "Sample vs FA")
    im_psn = plot_correlation_matrix(ax_psn, sample_corr, psn_corr, "Sample vs Poisson")
    im_com = plot_correlation_matrix(ax_com, sample_corr, com_corr, "Sample vs COM-Poisson")

    for ax, im in [(ax_fa, im_fa), (ax_psn, im_psn), (ax_com, im_com)]:
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
