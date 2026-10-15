"""Type definitions for the canonical correlation analysis example."""

from typing import TypedDict


class ViewFit(TypedDict):
    """One view split into the shared signal and the private noise: x = mean + signal * z + noise, z standard."""

    mean: list[float]
    signal: list[float]
    noise_cov: list[list[float]]


class CCAFit(TypedDict):
    """One fit of a two-view CCA model, by one training method."""

    method: str

    # Training
    log_likelihoods: list[float]

    # Each view of the fitted model, split into signal and noise (from model samples)
    fst_view: ViewFit
    snd_view: ViewFit

    # Latent recovery, after linear alignment to ground truth
    inferred_latents: list[list[float]]
    latent_rmse: float

    # The second view predicted from the first, E[y | x] under the fitted model
    predicted_snd: list[list[float]]
    prediction_rmse: float

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

    # Data (standardized) and ground truth
    observations: list[list[float]]
    true_latents: list[list[float]]
    true_fst_view: ViewFit
    true_snd_view: ViewFit
    true_cross_covariance: list[list[float]]

    # Fits
    gradient_ascent: CCAFit
    expectation_maximization: CCAFit
