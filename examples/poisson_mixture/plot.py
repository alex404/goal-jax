"""Plotting code for COM-Poisson mixture model analysis."""

from typing import Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

from ..shared import example_paths
from .types import CovarianceStatistics, PoissonAnalysisResults

# Type aliases for clarity
ModelType: TypeAlias = Literal["fan", "psn", "com", "grt", "sample"]

# Color schemes
MODEL_COLORS = {
    "fan": sns.color_palette("Set1")[0],  # Factor analysis
    "psn": sns.color_palette("Set1")[1],  # Poisson mixture
    "com": sns.color_palette("Set1")[2],  # COM-Poisson mixture
    "grt": sns.color_palette("Set1")[3],  # Ground truth
    "sample": sns.color_palette("Set1")[4],  # Sample statistics
}
CORRELATION_CMAP = sns.color_palette("RdBu", as_cmap=True)


def plot_means_comparison(
    ax: Axes,
    sample_stats: CovarianceStatistics,
    model_stats: CovarianceStatistics,
    model_type: ModelType,
) -> None:
    """Plot scatter comparison of sample vs model means.

    Parameters:
    -----------
    ax: Axes
        The matplotlib axes on which to plot
    sample_stats: CovarianceStatistics
        Statistics from the sample data
    model_stats: CovarianceStatistics
        Statistics from the model (fan, psn, or com)
    model_type: ModelType
        Type of model for color selection and labeling
    """
    sample_means = np.array(sample_stats["means"], dtype=np.float64)
    model_means = np.array(model_stats["means"], dtype=np.float64)

    # Calculate data ranges for consistent plotting
    data_min = min(float(sample_means.min()), float(model_means.min()))
    data_max = max(float(sample_means.max()), float(model_means.max()))
    padding = (data_max - data_min) * 0.05
    data_range = (data_min - padding, data_max + padding)

    # Plot scatter
    ax.scatter(
        model_means,
        sample_means,
        s=30,
        color=MODEL_COLORS[model_type],
        label=f"{model_type.upper()} vs Sample",
    )

    # Add diagonal line for perfect match reference
    ax.plot(data_range, data_range, "k--", alpha=0.5)

    # Styling
    ax.set_xlim(data_range)
    ax.set_ylim(data_range)
    ax.set_aspect("equal", "box")
    ax.set_ylabel("Sample Means")
    ax.set_xlabel(f"{model_type.upper()} Model Means")
    ax.legend(loc="best")


def plot_covariance_comparison(
    ax: Axes,
    sample_stats: CovarianceStatistics,
    model_stats: CovarianceStatistics,
    model_type: ModelType,
) -> None:
    """Plot scatter comparison of sample vs model covariances.

    Parameters:
    -----------
    ax: Axes
        The matplotlib axes on which to plot
    sample_stats: CovarianceStatistics
        Statistics from the sample data
    model_stats: CovarianceStatistics
        Statistics from the model (fan, psn, or com)
    model_type: ModelType
        Type of model for color selection and labeling
    """
    sample_covs = np.array(sample_stats["covariances"], dtype=np.float64)
    model_covs = np.array(model_stats["covariances"], dtype=np.float64)

    # Calculate data ranges
    data_min = min(float(sample_covs.min()), float(model_covs.min()))
    data_max = max(float(sample_covs.max()), float(model_covs.max()))
    padding = (data_max - data_min) * 0.05
    data_range = (data_min - padding, data_max + padding)

    # Plot scatter
    ax.scatter(
        model_covs,
        sample_covs,
        s=30,
        color=MODEL_COLORS[model_type],
        label=f"{model_type.upper()} vs Sample",
    )

    # Add diagonal line for perfect match reference
    ax.plot(data_range, data_range, "k--", alpha=0.5)

    # Styling
    ax.set_xlim(data_range)
    ax.set_ylim(data_range)
    ax.set_aspect("equal", "box")
    ax.set_ylabel("Sample Covariances")
    ax.set_xlabel(f"{model_type.upper()} Model Covariances")
    ax.legend(loc="best")


def plot_correlation_matrix(
    ax: Axes,
    correlations: NDArray[np.float64],
    title: str = "",
) -> AxesImage:
    """Plot correlation matrix with hierarchical clustering.

    Parameters:
    -----------
    ax: Axes
        The matplotlib axes on which to plot
    correlations: NDArray
        Correlation matrix to visualize
    title: str
        Optional title for the plot

    Returns:
    --------
    AxesImage
        The heatmap image object for colorbar attachment
    """
    # Hierarchical clustering to order matrix
    linkage_matrix = linkage(correlations, method="ward")
    _ = dendrogram(linkage_matrix, no_plot=True)
    order = leaves_list(linkage_matrix)

    # Reorder correlation matrix
    sorted_correlations = correlations[order, :][:, order]

    # Create heatmap
    heatmap = ax.imshow(
        sorted_correlations,
        cmap=CORRELATION_CMAP,
        interpolation="nearest",
        vmin=-1,
        vmax=1,
    )

    # Styling
    if title:
        ax.set_title(title)
    ax.set_xlabel("Observable Dimension")
    ax.set_ylabel("Observable Dimension")
    ax.set_xticks(np.arange(sorted_correlations.shape[1]))
    ax.set_yticks(np.arange(sorted_correlations.shape[0]))
    ax.set_xticklabels(order)
    ax.set_yticklabels(order)

    return heatmap


def plot_split_correlation_matrix(
    ax: Axes,
    sample_corr: NDArray[np.float64],
    model_corr: NDArray[np.float64],
    model_label: str,
) -> AxesImage:
    """Plot split correlation matrix with sample in lower triangle and model in upper.

    Parameters:
    -----------
    ax: Axes
        The matplotlib axes on which to plot
    sample_corr: NDArray
        Sample correlation matrix
    model_corr: NDArray
        Model correlation matrix
    model_label: str
        Label for the model (for title)

    Returns:
    --------
    AxesImage
        The heatmap image object for colorbar attachment
    """
    # Create copy of correlation matrix
    n = sample_corr.shape[0]
    combined_corr = np.zeros_like(sample_corr)

    # Fill lower triangle with sample correlations (including diagonal)
    for i in range(n):
        for j in range(i + 1):
            combined_corr[i, j] = sample_corr[i, j]

    # Fill upper triangle with model correlations (excluding diagonal)
    for i in range(n):
        for j in range(i + 1, n):
            combined_corr[i, j] = model_corr[i, j]

    # Create heatmap
    heatmap = ax.imshow(
        combined_corr,
        cmap=CORRELATION_CMAP,
        interpolation="nearest",
        vmin=-1,
        vmax=1,
    )

    # Styling
    ax.set_title(f"Sample (lower) vs {model_label} (upper)")
    ax.set_xlabel("Observable Dimension")
    ax.set_ylabel("Observable Dimension")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    return heatmap


def plot_training_history(ax: Axes, results: PoissonAnalysisResults) -> None:
    """Plot training history of log likelihoods for all models.

    Parameters:
    -----------
    ax: Axes
        The matplotlib axes on which to plot
    results: PoissonAnalysisResults
        Analysis results containing log likelihood sequences
    """
    # Styling
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Log Likelihood")

    # Plot each model's training history
    for key, label, color in [
        ("fan_lls", "Factor Analysis", MODEL_COLORS["fan"]),
        ("psn_lls", "Poisson Mixture", MODEL_COLORS["psn"]),
        ("com_lls", "COM-Poisson Mixture", MODEL_COLORS["com"]),
    ]:
        lls = cast(list[float], results[key])
        ax.plot(lls, label=label, color=color)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def main() -> None:
    """Load results and create plots according to the specified layout."""
    paths = example_paths(__file__)

    # Load analysis results
    results = cast(PoissonAnalysisResults, paths.load_analysis())

    # Convert correlation matrices to numpy arrays for easier handling
    sample_corr = np.array(
        results["sample_stats"]["correlation_matrix"], dtype=np.float64
    )
    fan_corr = np.array(results["fan_stats"]["correlation_matrix"], dtype=np.float64)
    psn_corr = np.array(results["psn_stats"]["correlation_matrix"], dtype=np.float64)
    com_corr = np.array(results["com_stats"]["correlation_matrix"], dtype=np.float64)

    # Create figure with specified layout
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # Create axes for each subplot
    # Top row
    ax_means = fig.add_subplot(gs[0, 0])
    ax_covs = fig.add_subplot(gs[0, 1])
    ax_train = fig.add_subplot(gs[0, 2])

    # Bottom row - correlation matrices
    ax_corr_fan = fig.add_subplot(gs[1, 0])
    ax_corr_psn = fig.add_subplot(gs[1, 1])
    ax_corr_com = fig.add_subplot(gs[1, 2])

    # Plot scatter comparisons with all models in the same plot
    # Means comparison
    for model_type, model_stats in [
        ("fan", results["fan_stats"]),
        ("psn", results["psn_stats"]),
        ("com", results["com_stats"]),
    ]:
        plot_means_comparison(
            ax_means, results["sample_stats"], model_stats, model_type
        )

    # Covariance comparison
    for model_type, model_stats in [
        ("fan", results["fan_stats"]),
        ("psn", results["psn_stats"]),
        ("com", results["com_stats"]),
    ]:
        plot_covariance_comparison(
            ax_covs, results["sample_stats"], model_stats, model_type
        )

    # Plot training history
    plot_training_history(ax_train, results)

    # Plot split correlation matrices
    img_fan = plot_split_correlation_matrix(
        ax_corr_fan, sample_corr, fan_corr, "Factor Analysis"
    )
    img_psn = plot_split_correlation_matrix(
        ax_corr_psn, sample_corr, psn_corr, "Poisson Mixture"
    )
    img_com = plot_split_correlation_matrix(
        ax_corr_com, sample_corr, com_corr, "COM-Poisson Mixture"
    )

    # Add colorbars
    plt.colorbar(img_fan, ax=ax_corr_fan)
    plt.colorbar(img_psn, ax=ax_corr_psn)
    plt.colorbar(img_com, ax=ax_corr_com)

    # Finalize layout
    fig.suptitle("COM-Poisson Mixture Model Analysis", fontsize=16)
    fig.tight_layout()

    # Save figure
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
