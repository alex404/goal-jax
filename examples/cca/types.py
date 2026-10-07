"""Type definitions for the canonical correlation analysis example."""

from typing import TypedDict


class CCAFit(TypedDict):
    """One fit of a two-view CCA model, by one training method."""

    method: str

    # Training
    log_likelihoods: list[float]

    # Latent recovery, after linear alignment to ground truth
    inferred_latents: list[list[float]]
    latent_rmse: float

    # Cross-view structure: the quantity CCA exists to capture
    learned_cross_covariance: list[list[float]]
    cross_covariance_error: float


class CCAResults(TypedDict):
    """Results from fitting a two-view CCA model by gradient ascent and by EM to the same data."""

    # Configuration
    fst_dim: int
    snd_dim: int
    lat_dim: int
    n_samples: int

    # Ground truth
    true_latents: list[list[float]]
    true_cross_covariance: list[list[float]]

    # Fits
    gradient_ascent: CCAFit
    expectation_maximization: CCAFit
