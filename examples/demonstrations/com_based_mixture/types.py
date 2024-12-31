"""Type definitions for mixture model comparison example."""

from typing import TypedDict


class MixtureComResults(TypedDict):
    """Results comparing factor analysis data fit with COM-Poisson mixture.

    Contains:
    - Ground truth parameters from factor analysis
    - Estimated parameters from COM mixture
    - Sample statistics from discretized data
    - Training metrics

    For each model/sample:
    - means: First moments (expected values)
    - fano_factors: Variance/mean ratios
    - covariances: Second moment structure
    - correlation_matrix: Normalized covariances
    """

    # Factor Analysis ground truth
    fa_means: list[float]
    fa_fano_factors: list[float]
    fa_covariances: list[float]
    fa_correlation_matrix: list[list[float]]

    # COM Mixture estimates
    cbm_means: list[float]
    cbm_fano_factors: list[float]
    cbm_covariances: list[float]
    cbm_correlation_matrix: list[list[float]]

    # Sample statistics
    sample_means: list[float]
    sample_fano_factors: list[float]
    sample_covariances: list[float]
    sample_correlation_matrix: list[list[float]]

    # Training history
    cbm_lls: list[float]  # Log-likelihood values during training
