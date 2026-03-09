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
    observations: list[list[float]]  # (n_steps, obs_dim)
    true_latents: list[list[float]]  # (n_steps+1, lat_dim)

    # Oracle inference (true parameters)
    filtered_means: list[list[float]]  # (n_steps, lat_dim)
    filtered_stds: list[list[float]]  # (n_steps, lat_dim)
    smoothed_means: list[list[float]]  # (n_steps, lat_dim)
    smoothed_stds: list[list[float]]  # (n_steps, lat_dim)

    # Learning curves
    log_likelihoods: dict[str, list[float]]  # {"Gradient": [...], "EM": [...]}

    # Learned inference — keyed by method name
    learned_filtered_means: dict[str, list[list[float]]]
    learned_filtered_stds: dict[str, list[list[float]]]
    learned_smoothed_means: dict[str, list[list[float]]]
    learned_smoothed_stds: dict[str, list[list[float]]]

    # Metrics
    true_log_lik: float
    initial_log_lik: float
