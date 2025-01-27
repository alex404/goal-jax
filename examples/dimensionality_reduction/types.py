# common.py
"""Common definitions for LGM model examples."""

from typing import TypedDict


class LGMResults(TypedDict):
    """Complete results for LGM analysis."""

    sample: list[list[float]]  # List of [x,y] points
    plot_bounds: tuple[float, float, float, float]  # Bounds for plotting
    training_lls: dict[str, list[float]]  # Log likelihoods during training
    grid_points: dict[str, list[list[float]]]  # Points where densities are evaluated
    ground_truth_densities: list[float]  # Ground truth densities
    initial_densities: dict[str, list[float]]  # Model densities in data space
    learned_densities: dict[str, list[float]]  # Model densities in data space
