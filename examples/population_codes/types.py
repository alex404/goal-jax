"""Result types for population code example."""

from typing import TypedDict


class PopulationCodeResults(TypedDict):
    """Results from population code inference."""

    # Tuning curves
    tuning_curve_grid: list[float]
    tuning_curves: list[list[float]]  # (n_neurons, n_grid)
    preferred_directions: list[float]

    # Regression diagnostics
    regression_grid: list[float]
    regression_actual: list[float]  # Actual sum of firing rates
    regression_fitted: list[float]  # Fitted regression values

    # Observations
    true_stimuli: list[float]
    spike_counts: list[list[int]]  # (n_samples, n_neurons)

    # Inference
    posterior_means: list[float]
    posterior_concentrations: list[float]

    # Error statistics
    mean_circular_error: float
