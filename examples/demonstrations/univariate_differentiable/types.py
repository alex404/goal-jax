"""Types for numerical univariate results."""

from typing import TypedDict


class VonMisesResults(TypedDict):
    """Results from fitting a von Mises distribution."""

    # Data
    samples: list[float]  # Generated/observed samples
    training_history: list[float]  # History of optimization

    # Evaluation
    eval_points: list[float]  # Points where density is evaluated
    true_density: list[float]  # Ground truth density (if synthetic)
    fitted_density: list[float]  # Fitted density using optimized parameters


class CoMPoissonResults(TypedDict):
    """Results from fitting a COM-Poisson distribution."""

    samples: list[int]  # Generated/observed samples
    training_history: list[float]  # History of optimization
    eval_points: list[int]  # Points where pmf is evaluated
    true_pmf: list[float]  # Ground truth probability mass function
    fitted_pmf: list[float]  # Fitted pmf using optimized parameters


class DifferentiableUnivariateResults(TypedDict):
    von_mises: VonMisesResults
    com_poisson: CoMPoissonResults
