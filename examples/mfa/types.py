"""Type definitions for Mixture of Factor Analyzers examples."""

from typing import TypedDict


class MFAResults(TypedDict):
    """Complete results for MFA analysis."""

    # Sample data
    observations: list[list[float]]  # (n_samples, 3)
    ground_truth_components: list[int]  # (n_samples,) - true component assignments

    # Grid coordinates for 2D marginal plots
    plot_range: list[float]  # Shared x/y coordinates for all marginals

    # Training metrics
    log_likelihoods: list[float]  # Training history (n_steps,)
    ground_truth_ll: float  # Ground truth log likelihood on sample

    # 2D marginal densities (conditioning on third dimension at 0)
    # Each is a 2D grid: (n_points, n_points)

    # Ground truth densities
    ground_truth_density_x1x2: list[list[float]]
    ground_truth_density_x1x3: list[list[float]]
    ground_truth_density_x2x3: list[list[float]]

    # Initial fitted densities
    initial_density_x1x2: list[list[float]]
    initial_density_x1x3: list[list[float]]
    initial_density_x2x3: list[list[float]]

    # Final fitted densities
    final_density_x1x2: list[list[float]]
    final_density_x1x3: list[list[float]]
    final_density_x2x3: list[list[float]]

    # Component responsibilities (posterior over components given data)
    ground_truth_responsibilities: list[list[float]]  # (n_samples, n_components)
    final_responsibilities: list[list[float]]  # (n_samples, n_components)
