"""Test script for mixtures of replicated COM-Poisson distributions."""

import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Natural,
    Point,
)
from goal.models import (
    FactorAnalysis,
)

from .types import MixtureComResults


def create_ground_truth_model(
    n_dim: int = 10, n_factors: int = 3
) -> tuple[FactorAnalysis, Point[Natural, FactorAnalysis]]:
    """Create ground truth factor analysis model with specified loadings."""
    # Similar to your Haskell ldngs initialization
    loadings = jnp.array(
        [
            [0.9, -0.1, 0.1, 0.4, -0.8, -0.5, -0.2, 0.3, 0.2, -0.6],  # First factor
            [0.1, 0.8, -0.1, -0.4, 0.3, 0.2, 0.5, -0.4, 0.2, 0.7],  # Second factor
            [0.1, 0.1, 0.7, -0.2, 0.8, 0.3, -0.3, 0.3, -0.6, 0.4],  # Third factor
        ]
    )
    # Create diagonal noise structure similar to your Haskell diag
    diag_values = ...

    fa_model = FactorAnalysis(n_dim, n_factors)
    ...
    return fa_model, params


def compute_mixture_results(
    key: Array,
    sample_size: int = 1000,
    n_steps: int = 100,
    n_epochs: int = 200,
) -> MixtureComResults:
    """Run complete analysis comparing FA to COM-mixture."""
    # 1. Create ground truth FA model
    fa_model, fa_params = create_ground_truth_model()

    # 2. Generate and discretize samples
    samples = fa_model.sample(key, fa_params, sample_size)
    count_samples = jnp.maximum(0, jnp.round(samples))

    # 3. Fit COM-mixture model
    ...
