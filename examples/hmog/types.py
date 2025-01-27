"""Common definitions for hierarchical mixture of Gaussians examples."""

from typing import TypedDict


class HMoGResults(TypedDict):
    """Complete results for hierarchical mixture of Gaussians analysis."""

    plot_range_x1: list[float]  # X1 coordinates for plotting
    plot_range_x2: list[float]  # X2 coordinates for plotting
    plot_range_y: list[float]  # Y coordinates for latent plotting
    observations: list[list[float]]  # List of [x1,x2] points
    log_likelihoods: dict[str, list[float]]  # Training log likelihoods
    observable_densities: dict[str, list[list[float]]]  # [true, init, final] densities
    mixture_densities: dict[str, list[float]]  # [true, init, final] latent densities
