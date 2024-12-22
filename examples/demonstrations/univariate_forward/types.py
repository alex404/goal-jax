"""Types for numerical univariate results."""

from typing import TypedDict


class VonMisesOptimizationStep(TypedDict):
    """Results from a single optimization step."""

    natural_params: list[float]  # Current natural parameters
    mean_params: list[float]  # Current mean parameters (from to_mean)
    target_means: list[float]  # Target mean parameters we're trying to match
    gradient: list[float]  # Current gradient
    loss: float  # Current loss value


class VonMisesResults(TypedDict):
    """Results from fitting a von Mises distribution."""

    # Ground truth parameters (if synthetic)
    true_mu: float  # True mean direction
    true_kappa: float  # True concentration

    # Data
    samples: list[float]  # Generated/observed samples

    # Optimization results
    optimization_path: list[VonMisesOptimizationStep]  # History of optimization
    final_natural_params: list[float]  # Final fitted natural parameters

    # Evaluation
    eval_points: list[float]  # Points where density is evaluated
    true_density: list[float]  # Ground truth density (if synthetic)
    fitted_density: list[float]  # Fitted density using optimized parameters


class ForwardUnivariateResults(TypedDict):
    """Container for all numerical univariate results."""

    von_mises: VonMisesResults
