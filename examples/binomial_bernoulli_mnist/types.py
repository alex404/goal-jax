"""Type definitions for Binomial-Bernoulli MNIST example."""

from typing import TypedDict


class ModelResults(TypedDict):
    """Results for a single model configuration."""

    # Training history
    training_elbos: list[float]
    conjugation_errors: list[float]  # Var[f̃] over time
    conjugation_stds: list[float]  # Std[f̃] over time
    conjugation_r2s: list[float]  # R² over time
    rho_norms: list[float]  # Norm of rho parameters over time

    # Clustering metrics
    cluster_purity: float
    nmi: float  # Normalized Mutual Information
    cluster_counts: list[int]
    cluster_assignments: list[int]

    # Reconstruction
    reconstruction_error: float

    # Model-specific visualizations
    generated_samples: list[list[float]]  # Samples from generative model
    cluster_prototypes: list[list[float]]  # Mean image per cluster


class BinomialBernoulliMNISTResults(TypedDict):
    """Results from Binomial-Bernoulli Mixture MNIST training and analysis."""

    # Comparison of three models
    learnable_rho: ModelResults
    fixed_rho: ModelResults
    conjugate_regularized: ModelResults  # Learnable rho with conjugation penalty

    # Shared data
    true_labels: list[int]
    original_images: list[list[float]]
    reconstructed_images: list[list[float]]  # From best model
