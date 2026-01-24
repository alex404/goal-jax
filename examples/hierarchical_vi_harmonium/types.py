"""Type definitions for Hierarchical VI Harmonium results."""

from typing import TypedDict


class HierarchicalVIConfig(TypedDict):
    """Configuration for hierarchical variational inference training."""

    n_mc_samples: int
    kl_weight: float
    learning_rate_harmonium: float
    learning_rate_rho: float
    n_clusters: int
    n_latent: int


class HierarchicalVIResults(TypedDict):
    """Results from Hierarchical VI Harmonium training."""

    # Training metrics over epochs
    elbo_history: list[float]
    reconstruction_history: list[float]
    kl_history: list[float]
    rho_norm_history: list[float]

    # Clustering metrics
    cluster_assignments: list[int]  # Predicted cluster for each test sample
    true_labels: list[int]  # True digit labels
    cluster_purity: float  # Cluster purity score
    cluster_counts: list[int]  # Number of samples per cluster

    # Final model outputs
    weight_filters: list[list[list[float]]]  # Learned weight filters
    reconstructed_images: list[list[float]]  # Reconstructions
    generated_samples: list[list[float]]  # Samples from the model
    cluster_prototypes: list[list[float]]  # Mean image per cluster

    # Per-cluster samples (list of n_clusters lists of images)
    per_cluster_samples: list[list[list[float]]]

    # Shared data
    original_images: list[list[float]]

    # Configuration
    config: HierarchicalVIConfig
