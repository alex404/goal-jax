"""Common definitions for Gaussian-Boltzmann harmonium examples."""

from typing import TypedDict


class GaussianBoltzmannResults(TypedDict):
    """Results from Gaussian-Boltzmann harmonium analysis."""

    observations: list[list[float]]  # Training data points
    plot_range_x: list[float]  # X coordinates for density plotting
    plot_range_y: list[float]  # Y coordinates for density plotting
    learned_density: list[list[float]]  # Observable density heatmap
    log_likelihoods: list[float]  # Training log likelihoods
    component_confidence_ellipses: list[list[list[float]]]  # Confidence ellipses per component
    prior_moment_matrix: list[list[float]]  # Prior moment matrix
    posterior_observations: list[list[float]]  # Observations used for posterior
    posterior_moment_matrices: list[list[list[float]]]  # Posterior moment matrices
