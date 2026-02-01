"""Type definitions for dimensionality reduction example."""

from typing import TypedDict


class TrajectoryResults(TypedDict):
    """Results from latent trajectory recovery experiment."""

    # Ground truth
    true_latents: list[list[float]]  # (n_points, 2) - true latent positions
    observations: list[list[float]]  # (n_points, obs_dim) - noisy observations

    # Model results
    training_lls: list[float]  # Log-likelihoods during EM
    recovered_latents: list[list[float]]  # (n_points, 2) - inferred latent positions
    aligned_latents: list[list[float]]  # (n_points, 2) - Procrustes-aligned to true

    # Reconstruction
    reconstructions: list[list[float]]  # (n_points, obs_dim) - reconstructed observations
    reconstruction_errors: list[float]  # Per-point reconstruction error

    # Configuration
    obs_dim: int
    latent_dim: int
    n_points: int
    noise_std: float
