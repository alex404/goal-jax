"""Common definitions for mixture model examples."""

from typing import TypedDict


class MixtureResults(TypedDict):
    """Complete results for mixture model analysis."""

    sample: list[list[float]]  # List of [x,y] points
    plot_xs: list[list[float]]  # 2D grid of x coordinates
    plot_ys: list[list[float]]  # 2D grid of y coordinates
    scipy_densities: list[list[float]]  # Scipy density on grid
    ground_truth_densities: list[list[float]]  # Ground truth density on grid
    init_positive_definite_densities: list[list[float]]  # PD model density on grid
    init_diagonal_densities: list[list[float]]  # Diagonal() model density on grid
    init_isotropic_densities: list[list[float]]  # Isotropic model density on grid
    positive_definite_densities: list[list[float]]  # PD model density on grid
    diagonal_densities: list[list[float]]  # Diagonal() model density on grid
    isotropic_densities: list[list[float]]  # Isotropic model density on grid
    ground_truth_ll: float
    training_lls: dict[
        str, list[float]
    ]  # Log likelihoods during training for each model
