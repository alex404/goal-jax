"""Plotting code for COM-Poisson mixture model analysis."""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

from ...shared import initialize_paths
from .types import ComAnalysisResults

# Color schemes
scatter_palette = sns.color_palette("Set1", n_colors=2)
correlation_palette = sns.color_palette("RdBu", as_cmap=True)


def plot_mean_scatter(
    ax: Axes,
    smp_means: NDArray[np.float64],
    cbm_means: NDArray[np.float64],
) -> None:
    """Plot scatter comparison of sample vs fitted means."""
    data_min = min(float(smp_means.min()), float(cbm_means.min()))
    data_max = max(float(smp_means.max()), float(cbm_means.max()))
    padding = (data_max - data_min) * 0.05
    data_range = (data_min - padding, data_max + padding)

    ax.scatter(cbm_means, smp_means, s=10, color=scatter_palette[0])
    ax.set_xlim(data_range)
    ax.set_ylim(data_range)
    ax.set_aspect("equal", "box")
    ax.set_ylabel("Sample Means")
    ax.set_xlabel("COM Mixture Means")


def plot_covariance_scatter(
    ax: Axes,
    smp_covs: NDArray[np.float64],
    cbm_covs: NDArray[np.float64],
) -> None:
    """Plot scatter comparison of sample vs fitted covariances."""
    data_min = min(float(smp_covs.min()), float(cbm_covs.min()))
    data_max = max(float(smp_covs.max()), float(cbm_covs.max()))
    padding = (data_max - data_min) * 0.05
    data_range = (data_min - padding, data_max + padding)

    ax.scatter(cbm_covs, smp_covs, s=10, color=scatter_palette[1])
    ax.set_xlim(data_range)
    ax.set_ylim(data_range)
    ax.set_aspect("equal", "box")
    ax.set_ylabel("Sample Covariances")
    ax.set_xlabel("COM Mixture Covariances")


def plot_correlation_matrix(
    ax: Axes,
    correlations: NDArray[np.float64],
) -> AxesImage:
    """Plot correlation matrix with hierarchical clustering."""
    # Hierarchical clustering to order matrix
    linkage_matrix = linkage(correlations, method="ward")
    _ = dendrogram(linkage_matrix, no_plot=True)
    order = leaves_list(linkage_matrix)

    # Reorder correlation matrix
    sorted_correlations = correlations[order, :][:, order]

    # Create heatmap
    heatmap = ax.imshow(
        sorted_correlations,
        cmap=correlation_palette,
        interpolation="nearest",
        vmin=-1,
        vmax=1,
    )

    ax.set_xlabel("Observable Dimension")
    ax.set_ylabel("Observable Dimension")
    ax.set_xticks(np.arange(sorted_correlations.shape[1]))
    ax.set_yticks(np.arange(sorted_correlations.shape[0]))
    ax.set_xticklabels(order)
    ax.set_yticklabels(order)

    return heatmap


def plot_training_history(ax: Axes, lls: list[float]) -> None:
    """Plot training history of log likelihoods."""
    steps = np.arange(len(lls))
    ax.plot(steps, lls)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Log Likelihood")
    ax.grid(True)


def create_com_mixture_plots(results: ComAnalysisResults) -> Figure:
    """Create figure with all COM mixture analysis plots."""
    # Create figure layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    # Create axes
    ax_means = fig.add_subplot(gs[0, 0])
    ax_covs = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[1, 0])
    ax_train = fig.add_subplot(gs[1, 1])

    # Plot comparisons
    plot_mean_scatter(
        ax_means,
        np.array(results["fa_stats"]["means"], dtype=np.float64),
        np.array(results["cbm_stats"]["means"], dtype=np.float64),
    )

    plot_covariance_scatter(
        ax_covs,
        np.array(results["fa_stats"]["covariances"], dtype=np.float64),
        np.array(results["cbm_stats"]["covariances"], dtype=np.float64),
    )

    heatmap = plot_correlation_matrix(
        ax_corr, np.array(results["cbm_stats"]["correlation_matrix"], dtype=np.float64)
    )
    fig.colorbar(heatmap, ax=ax_corr)

    plot_training_history(ax_train, results["training_lls"])

    fig.tight_layout()
    return fig


def main() -> None:
    """Load results and create plots."""
    paths = initialize_paths(__file__)

    # Load results
    with open(paths.analysis_path) as f:
        results = cast(ComAnalysisResults, json.load(f))

    # sort
    print("Sorted FA Covariances:")
    print(np.sort(np.array(results["fa_stats"]["covariances"])))
    print("Sorted Discrete Covariances:")
    print(np.sort(np.array(results["sample_stats"]["covariances"])))
    print("Sorted CBM Covariances:")
    print(np.sort(np.array(results["cbm_stats"]["covariances"])))
    # Create and save figure
    fig = create_com_mixture_plots(results)
    fig.savefig(
        paths.results_dir / "com_mixture_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
