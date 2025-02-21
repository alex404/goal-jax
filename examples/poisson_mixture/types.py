"""Type definitions for COM-Poisson mixture model analysis."""

from typing import TypedDict


class CovarianceStatistics(TypedDict):
    """Statistics comparing model covariance structures."""

    means: list[float]  # Mean for each dimension
    fano_factors: list[float]  # Variance/mean ratios
    covariances: list[float]  # Lower triangular elements
    correlation_matrix: list[list[float]]  # Full correlation matrices


class PoissonAnalysisResults(TypedDict):
    """Complete results for COM-Poisson mixture model analysis.

    Contains:
    1. Factor Analysis ground truth model statistics
    2. COM Mixture fitted model statistics
    3. Sample empirical statistics
    4. Training history
    """

    grt_stats: CovarianceStatistics  # Ground truth factor analysis
    sample_stats: CovarianceStatistics  # Empirical statistics
    fan_stats: CovarianceStatistics  # Fitted factor analysis
    psn_stats: CovarianceStatistics  # Fitted Poisson mixture
    com_stats: CovarianceStatistics  # Fitted COM-Poisson mixture

    fan_lls: list[float]  # Factor analysis log-likelihoods
    psn_lls: list[float]  # Poisson mixture log-likelihoods
    com_lls: list[float]  # Training log-likelihoods
