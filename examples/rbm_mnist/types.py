"""Type definitions for RBM MNIST example."""

from typing import TypedDict


class RBMResults(TypedDict):
    """Results from RBM training on MNIST."""

    # Training metrics
    reconstruction_errors: list[float]
    free_energies: list[float]

    # Learned filters (n_hidden x 28 x 28)
    weight_filters: list[list[list[float]]]

    # Sample reconstructions
    original_images: list[list[float]]  # (n_samples, 784)
    reconstructed_images: list[list[float]]  # (n_samples, 784)

    # Generated samples from Gibbs chain
    generated_samples: list[list[float]]  # (n_samples, 784)
