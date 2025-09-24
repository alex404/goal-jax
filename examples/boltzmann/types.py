from typing import TypedDict


class BoltzmannPatternResults(TypedDict):
    """Complete results for Boltzmann pattern learning analysis."""

    # Ground truth patterns
    ground_truth_patterns: list[list[int]]  # List of binary patterns
    pattern_frequencies: list[float]  # Frequency of each pattern in training data

    # Fitted model parameters
    fitted_bias: list[float]  # Learned bias parameters
    fitted_interactions: list[list[float]]  # Learned interaction matrix

    # Model evaluation
    all_possible_states: list[list[int]]  # All 2^n possible binary states
    true_probabilities: list[float]  # Ground truth probabilities
    learned_probabilities: list[float]  # Learned model probabilities
    energy_values: list[float]  # Energy of each state under learned model

    # Generated samples
    generated_samples: list[list[int]]  # Samples from fitted model
    generated_frequencies: list[float]  # Frequency of each pattern in samples

    # Training data
    training_losses: list[float]  # Loss values during optimization

    # Sampling convergence test
    sampling_sizes: list[int]  # Sample sizes for convergence test
    sampling_thin_levels: list[int]  # Thinning levels (Gibbs steps between samples)
    sampling_errors_matrix: list[list[float]]  # L2 errors [sample_size][thin_level]

    # Visualization data
    pattern_grid_size: int  # Size of 2D grid (assuming square patterns)
    state_labels: list[str]  # String representation of each state