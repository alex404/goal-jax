"""Type definitions for Mixture of Factor Analyzers examples."""

from typing import TypedDict


class MFAResults(TypedDict):
    """Complete results for MFA analysis.

    Compares two models:
    - FA (Factor Analysis): FactorAnalysis base with full latent covariance
    - Diag: NormalLGM[Diagonal, Diagonal] base with diagonal latent covariance
    """

    # Sample data
    observations: list[list[float]]  # (n_samples, 3)
    ground_truth_components: list[int]  # (n_samples,) - true component assignments

    # Grid coordinates for 2D marginal plots
    plot_range: list[float]  # Shared x/y coordinates for all marginals

    # Training metrics (keyed by model name)
    log_likelihoods: dict[str, list[float]]  # {"FA": [...], "Diag": [...]}
    ground_truth_ll: float  # Log-likelihood achieved by ground truth model

    # 2D marginal densities (conditioning on third dimension at 0)
    # Each is a 2D grid: (n_points, n_points)
    # Keyed by: "Ground Truth", "FA Initial", "FA Final", "Diag Initial", "Diag Final"
    density_x1x2: dict[str, list[list[float]]]
    density_x1x3: dict[str, list[list[float]]]
    density_x2x3: dict[str, list[list[float]]]

    # Posterior assignments (posterior probabilities over components given data)
    ground_truth_assignments: list[list[float]]  # (n_samples, n_components)
    final_assignments: dict[str, list[list[float]]]  # {"FA": [...], "Diag": [...]}
