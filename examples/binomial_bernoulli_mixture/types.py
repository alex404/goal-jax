"""Type definitions for Binomial-Bernoulli Mixture example."""

from typing import TypedDict


class BinomialBernoulliResults(TypedDict):
    """Results from Binomial-Bernoulli Mixture training and analysis."""

    # Data
    test_data: list[list[float]]
    test_labels: list[int]

    # Model predictions
    predicted_labels: list[int]
    cluster_probs: list[list[float]]
    reconstructions: list[list[float]]

    # Training history
    training_elbos: list[float]

    # Final metrics
    cluster_purity: float
    reconstruction_error: float
    cluster_distribution: list[int]
