"""Type declarations for HMM example results."""

from typing import TypedDict


class HMMResults(TypedDict):
    """Results from HMM learning example."""

    # Model configuration
    n_obs: int
    n_states: int
    n_steps: int
    n_sequences: int

    # Sample trajectory (first sequence) -- integer indices
    observations: list[int]
    true_states: list[int]

    # Oracle inference (true parameters): per-step probability over latent states
    filtered_probs: list[list[float]]
    smoothed_probs: list[list[float]]

    # Learning curves
    log_likelihoods: dict[str, list[float]]

    # Learned inference -- keyed by method name
    learned_filtered_probs: dict[str, list[list[float]]]
    learned_smoothed_probs: dict[str, list[list[float]]]

    # Metrics
    true_log_lik: float
    initial_log_lik: float
