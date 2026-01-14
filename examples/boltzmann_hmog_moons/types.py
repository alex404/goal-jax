"""Common definitions for Boltzmann HMoG two-moons example."""

from typing import TypedDict


class BoltzmannHMoGMoonsResults(TypedDict):
    """Results from Boltzmann HMoG two-moons analysis."""

    # Data
    observations: list[list[float]]  # Training data points [x, y]
    moon_labels: list[int]  # Ground truth moon assignment (0 or 1)

    # Plot ranges
    plot_range_x: list[float]  # X coordinates for density plotting
    plot_range_y: list[float]  # Y coordinates for density plotting

    # Learned densities (2D heatmaps)
    gmm_density: list[list[float]]  # Gaussian mixture baseline
    boltzmann_density: list[list[float]]  # SymmetricBoltzmannHMoG

    # Training histories
    gmm_log_likelihoods: list[float]
    boltzmann_log_likelihoods: list[float]

    # Cluster assignments (soft probabilities per observation)
    gmm_assignments: list[list[float]]  # [n_obs, n_components]
    boltzmann_assignments: list[list[float]]

    # Binary activations for Boltzmann model (posterior means)
    boltzmann_binary_activations: list[list[float]]  # [n_obs, n_latents]
