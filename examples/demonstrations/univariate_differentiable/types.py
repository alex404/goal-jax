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


class DifferentiableUnivariateResults(TypedDict):
    """Container for all numerical univariate results."""

    von_mises: VonMisesResults
