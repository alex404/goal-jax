"""Type definitions for the canonical correlation analysis example."""

from typing import TypedDict


class CCAResults(TypedDict):
    """Results from fitting a two-view CCA model to data from a known one."""

    # Configuration
    fst_dim: int
    snd_dim: int
    lat_dim: int
    n_samples: int
    n_steps: int

    # Training
    log_likelihoods: list[float]

    # Latent recovery, after Procrustes alignment to ground truth
    true_latents: list[list[float]]
    inferred_latents: list[list[float]]
    latent_rmse: float

    # Cross-view structure: the quantity CCA exists to capture
    true_cross_covariance: list[list[float]]
    learned_cross_covariance: list[list[float]]
    cross_covariance_error: float
