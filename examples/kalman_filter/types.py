"""Type declarations for Kalman filter example results."""

from typing import TypedDict


class KalmanFilterResults(TypedDict):
    """Results from Kalman filter learning example."""

    # Model configuration
    obs_dim: int
    lat_dim: int
    n_steps: int
    n_sequences: int

    # Training data (one sample trajectory for visualization)
    observations: list[list[float]]
    true_latents: list[list[float]]

    # Oracle inference (true parameters)
    filtered_means: list[list[float]]
    filtered_stds: list[list[float]]
    smoothed_means: list[list[float]]
    smoothed_stds: list[list[float]]

    # Learning curves
    log_likelihoods: dict[str, list[float]]

    # Learned inference -- keyed by method name
    learned_filtered_means: dict[str, list[list[float]]]
    learned_filtered_stds: dict[str, list[list[float]]]
    learned_smoothed_means: dict[str, list[list[float]]]
    learned_smoothed_stds: dict[str, list[list[float]]]

    # Metrics
    true_log_lik: float
    initial_log_lik: float
