"""Result types for Boltzmann CD validation example."""

from typing import TypedDict


class BoltzmannCDResults(TypedDict):
    """Results comparing exact vs CD training."""

    # Training data
    observations: list[list[float]]

    # Density grid coordinates
    plot_range_x: list[float]
    plot_range_y: list[float]

    # NLL trajectories (per epoch)
    exact_nlls: list[float]
    cd_nlls: list[float]

    # Density grids
    initial_density: list[list[float]]
    exact_density: list[list[float]]
    cd_density: list[list[float]]
