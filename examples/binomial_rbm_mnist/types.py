# ruff: noqa: RUF003
"""Type definitions for Binomial RBM MNIST example."""

from typing import TypedDict


class ConjugationConfig(TypedDict):
    """Configuration for approximate conjugation."""

    alpha_cd: float
    alpha_conj: float
    n_conj_samples: int
    n_gibbs_conj: int


class BinomialRBMResults(TypedDict):
    """Results from Binomial RBM training on grayscale MNIST.

    Compares baseline CD training with conjugation-regularized training.
    """

    # Baseline results
    baseline_reconstruction_errors: list[float]
    baseline_free_energies: list[float]
    baseline_weight_filters: list[list[list[float]]]
    baseline_reconstructed_images: list[list[float]]
    baseline_generated_samples: list[list[float]]

    # Conjugated results
    conj_reconstruction_errors: list[float]
    conj_free_energies: list[float]
    conj_weight_filters: list[list[list[float]]]
    conj_reconstructed_images: list[list[float]]
    conj_generated_samples: list[list[float]]

    # Conjugation-specific metrics
    conjugation_errors: list[float]  # Estimated D(Q_rho || Q_Z) over training
    rho_norms: list[float]  # ||rho|| over training

    # Shared data
    original_images: list[list[float]]

    # Configuration
    config: ConjugationConfig
