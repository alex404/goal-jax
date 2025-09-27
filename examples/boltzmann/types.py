from typing import TypedDict


class BoltzmannPatternResults(TypedDict):
    """Complete results for Boltzmann pattern learning analysis."""

    # Model evaluation
    all_possible_states: list[list[int]]  # All 2^n possible binary states
    true_probabilities: list[float]  # Ground truth probabilities
    learned_probabilities: list[float]  # Learned model probabilities
    energy_values: list[float]  # Energy of each state under learned model

    # Training data
    training_losses: list[float]  # Loss values during optimization

    # Sampling convergence test
    sampling_sizes: list[int]  # Sample sizes for convergence test
    sampling_thin_levels: list[int]  # Thinning levels (Gibbs steps between samples)
    sampling_errors_matrix: list[list[float]]  # L2 errors [sample_size][thin_level]

    # Visualization data
    state_labels: list[str]  # String representation of each state
