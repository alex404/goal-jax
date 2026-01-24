"""Type definitions for Structured VI Harmonium results."""

from typing import TypedDict


class StructuredVIConfig(TypedDict):
    """Configuration for structured variational inference training."""

    n_mc_samples: int
    kl_weight: float
    learning_rate: float


class StructuredVIResults(TypedDict):
    """Results from Structured VI Harmonium training."""

    # Training metrics over epochs
    elbo_history: list[float]
    reconstruction_history: list[float]
    kl_history: list[float]
    rho_norm_history: list[float]
    conjugation_error_history: list[float]

    # Final model outputs
    weight_filters: list[list[list[float]]]
    reconstructed_images: list[list[float]]
    generated_samples: list[list[float]]

    # Comparison with CD baseline
    baseline_reconstruction_errors: list[float]
    baseline_free_energies: list[float]
    baseline_weight_filters: list[list[list[float]]]
    baseline_reconstructed_images: list[list[float]]
    baseline_generated_samples: list[list[float]]

    # Shared data
    original_images: list[list[float]]

    # Configuration
    config: StructuredVIConfig
