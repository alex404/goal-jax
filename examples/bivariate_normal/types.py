from typing import TypedDict


class BivariateResults(TypedDict):
    """Complete results for bivariate analysis."""

    sample: list[list[float]]  # List of [x,y] points
    plot_xs: list[list[float]]  # 2D grid of x coordinates
    plot_ys: list[list[float]]  # 2D grid of y coordinates
    scipy_densities: list[list[float]]
    ground_truth_densities: list[list[float]]
    positive_definite_densities: list[list[float]]
    diagonal_densities: list[list[float]]
    scale_densities: list[list[float]]
