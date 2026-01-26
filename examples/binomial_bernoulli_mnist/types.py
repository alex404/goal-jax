"""Type definitions for Binomial-Bernoulli MNIST example."""

from typing import TypedDict


class ModelResults(TypedDict):
    """Results for a single model configuration."""

    # Training history
    training_elbos: list[float]
    conjugation_errors: list[float]  # Variance of reduced learning signal over time
    rho_norms: list[float]  # Norm of rho parameters over time

    # Clustering metrics
    cluster_purity: float
    nmi: float  # Normalized Mutual Information
    cluster_counts: list[int]
    cluster_assignments: list[int]

    # Reconstruction
    reconstruction_error: float


class BinomialBernoulliMNISTResults(TypedDict):
    """Results from Binomial-Bernoulli Mixture MNIST training and analysis."""

    # Comparison of three models
    learnable_rho: ModelResults
    fixed_rho: ModelResults
    conjugate_regularized: ModelResults  # Learnable rho with conjugation penalty

    # Shared data
    true_labels: list[int]

    # Visualization data (from best model)
    original_images: list[list[float]]
    reconstructed_images: list[list[float]]
    cluster_prototypes: list[list[float]]  # Mean image per cluster
    generated_samples: list[list[float]]  # Samples from generative model
