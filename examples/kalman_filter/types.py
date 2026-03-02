"""Type declarations for Kalman filter example results."""

from typing import TypedDict


class KalmanFilterResults(TypedDict):
    """Results from Kalman filter learning example."""

    # Model configuration
    obs_dim: int
    lat_dim: int
    n_steps: int
    n_sequences: int
    n_em_steps: int

    # Training data (one sample trajectory for visualization)
    observations: list[list[float]]  # (n_steps, obs_dim)
    true_latents: list[list[float]]  # (n_steps+1, lat_dim)

    # Oracle inference (true parameters)
    filtered_means: list[list[float]]  # (n_steps, lat_dim)
    filtered_stds: list[list[float]]  # (n_steps, lat_dim)
    smoothed_means: list[list[float]]  # (n_steps, lat_dim)
    smoothed_stds: list[list[float]]  # (n_steps, lat_dim)

    # Learned inference (learned parameters)
    learned_filtered_means: list[list[float]]  # (n_steps, lat_dim)
    learned_filtered_stds: list[list[float]]  # (n_steps, lat_dim)
    learned_smoothed_means: list[list[float]]  # (n_steps, lat_dim)
    learned_smoothed_stds: list[list[float]]  # (n_steps, lat_dim)

    # Learning curve
    log_likelihoods: list[float]

    # Metrics
    true_log_lik: float
    initial_log_lik: float
    final_log_lik: float
