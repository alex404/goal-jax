"""Hierarchical Mixture of Gaussians (HMoG) models.

This module provides concrete implementations of hierarchical Gaussian models that combine
linear Gaussian dimensionality reduction with Gaussian mixture clustering, enabling joint
learning of latent factor representations and cluster assignments.

**Model structure**: HMoG models have two levels:

- **Lower harmonium**: Maps observations :math:`X \\in \\mathbb{R}^p` to first-level latent factors
  :math:`Y \\in \\mathbb{R}^d` using a linear Gaussian relationship (factor analysis/PCA)
- **Upper harmonium**: Models a mixture of Gaussians over the latent space :math:`Y`

The joint distribution factors as:

.. math::

    p(X, Y, Z) = p(Z) \\cdot p(Y | Z) \\cdot p(X | Y)

where :math:`Z \\in \\{1,\\ldots,K\\}` are discrete cluster assignments.

**Variants**: Three implementations with different analytical properties:

- **DifferentiableHMoG**: Gradient-based optimization, uses restricted posterior covariance
  for efficiency (e.g., diagonal)
- **SymmetricHMoG**: Symmetric posterior/prior structure, additional functionality like
  `join_conjugated`, but slower due to full covariance matrix operations
- **AnalyticHMoG**: Fully analytic, enables closed-form EM and bidirectional parameter conversion

Factory functions (`differentiable_hmog`, `symmetric_hmog`, `analytic_hmog`) provide
convenient construction for common configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticHierarchical,
    DifferentiableHierarchical,
    PositiveDefinite,
    SymmetricHierarchical,
)
from ..base.gaussian.normal import Normal
from ..harmonium.lgm import (
    NormalAnalyticLGM,
    NormalCovarianceEmbedding,
    NormalLGM,
)
from ..harmonium.mixture import AnalyticMixture, Mixture

### HMoG Classes ###


@dataclass(frozen=True)
class DifferentiableHMoG(
    DifferentiableHierarchical[NormalLGM, AnalyticMixture[Normal], Mixture[Normal]]
):
    """Differentiable Hierarchical Mixture of Gaussians.

    This model combines:
    1. A linear Gaussian model (factor analysis) mapping observations to latents
    2. A Gaussian mixture model over the latent space

    Supports gradient-based optimization via log-likelihood descent.
    Uses full covariance Gaussians in the latent space.

    **Posterior vs Prior Structure**: The posterior latent mixture (`pst_upr_hrm`) uses an
    AnalyticMixture with a restricted covariance structure for computational efficiency.
    The prior latent mixture (`prr_upr_hrm`) embeds the restricted structure into full
    covariance for conjugation parameter computation.
    """

    # HMoG-specific methods

    def posterior_categorical(self, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates.

        Returns the natural parameters of the categorical distribution over
        mixture components in the latent space given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components-1,) with categorical natural parameters
        """
        # Compute posterior latent mixture
        posterior = self.posterior_at(params, x)

        # Extract categorical marginal from the mixture
        return self.pst_upr_hrm.prior(posterior)

    def posterior_soft_assignments(self, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x).

        Returns the posterior probability distribution over mixture components
        in the latent space given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components,) giving p(z_k|x) for each component k
        """
        cat_natural = self.posterior_categorical(params, x)
        cat_means = self.pst_upr_hrm.lat_man.to_mean(cat_natural)
        return self.pst_upr_hrm.lat_man.to_probs(cat_means)

    def posterior_hard_assignment(self, params: Array, x: Array) -> Array:
        """Compute hard posterior assignments p(Z|x).

        Returns the index of the most probable mixture component
        in the latent space given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Integer index of the most probable component
        """
        soft_assignments = self.posterior_soft_assignments(params, x)
        return jnp.argmax(soft_assignments)


@dataclass(frozen=True)
class SymmetricHMoG(SymmetricHierarchical[NormalAnalyticLGM, Mixture[Normal]]):
    """Symmetric Hierarchical Mixture of Gaussians.

    This model supports gradient-based optimization with additional functionality
    (e.g., join_conjugated) not available in DifferentiableHMoG.

    The symmetric structure means the posterior and conjugated latent spaces are
    the same, enabling bidirectional parameter transformations.

    Trade-off: Matrix inversions happen in the space of full covariance matrices
    over the latent space, which can be slower than DifferentiableHMoG.
    """

    # HMoG-specific methods

    def posterior_categorical(self, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates.

        Returns the natural parameters of the categorical distribution over
        mixture components in the latent space given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components-1,) with categorical natural parameters
        """
        # Compute posterior latent mixture
        posterior = self.posterior_at(params, x)

        # Extract categorical marginal from the mixture
        return self.upr_hrm.prior(posterior)

    def posterior_assignments(self, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x).

        Returns the posterior probability distribution over mixture components
        in the latent space given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components,) giving p(z_k|x) for each component k
        """
        cat_natural = self.posterior_categorical(params, x)
        return self.upr_hrm.lat_man.to_probs(cat_natural)


@dataclass(frozen=True)
class AnalyticHMoG(AnalyticHierarchical[NormalAnalyticLGM, AnalyticMixture[Normal]]):
    """Analytic Hierarchical Mixture of Gaussians.

    This model enables:
    - Closed-form EM algorithm for learning (from AnalyticConjugated)
    - Bidirectional parameter conversion (mean ↔ natural)
    - Full analytical tractability

    Requires full covariance Gaussians in the latent space.
    """

    # HMoG-specific methods

    def posterior_categorical(self, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates.

        Returns the natural parameters of the categorical distribution over
        mixture components in the latent space given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components-1,) with categorical natural parameters
        """
        # Compute posterior latent mixture
        posterior = self.posterior_at(params, x)

        # Extract categorical marginal from the mixture
        return self.upr_hrm.prior(posterior)

    def posterior_assignments(self, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x).

        Returns the posterior probability distribution over mixture components
        in the latent space given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components,) giving p(z_k|x) for each component k
        """
        cat_natural = self.posterior_categorical(params, x)
        return self.upr_hrm.lat_man.to_probs(cat_natural)


## Factory Functions ##


def differentiable_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    lat_dim: int,
    pst_rep: PositiveDefinite,
    n_components: int,
) -> DifferentiableHMoG:
    """Create a differentiable hierarchical mixture of Gaussians model.

    This function constructs a hierarchical model combining:
    1. A bottom layer with a linear Gaussian model reducing observables to first-level latents
    2. A top layer with a Gaussian mixture model for modelling the latent distribution

    This model supports optimization via log-likelihood gradient descent.
    Uses full covariance Gaussians in the latent space.
    """

    pst_y_man = Normal(lat_dim, pst_rep)
    prr_y_man = Normal(lat_dim, PositiveDefinite())
    lwr_hrm = NormalLGM(obs_dim, obs_rep, lat_dim, pst_rep)
    mix_sub = NormalCovarianceEmbedding(pst_y_man, prr_y_man)
    pst_upr_hrm = AnalyticMixture(pst_y_man, n_components)

    # Use the embedding directly for the mixture
    prr_upr_hrm = Mixture(n_components, mix_sub)

    return DifferentiableHMoG(
        lwr_hrm,
        pst_upr_hrm,
        prr_upr_hrm,
    )


def symmetric_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    lat_dim: int,
    lat_rep: PositiveDefinite,
    n_components: int,
) -> SymmetricHMoG:
    """Create a symmetric hierarchical mixture of Gaussians model.

    Supports optimization via log-likelihood gradient descent with additional functionality
    (e.g., `join_conjugated`) not available in `DifferentiableHMoG`. The symmetric structure
    means posterior and prior use the same latent parameterization.

    Trade-off: Matrix inversions happen in the space of full covariance matrices over the
    latent space, which can be slower than `DifferentiableHMoG`.
    """

    mid_lat_man = Normal(lat_dim, PositiveDefinite())
    sub_lat_man = Normal(lat_dim, lat_rep)
    mix_sub = NormalCovarianceEmbedding(sub_lat_man, mid_lat_man)
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)

    # Use the embedding directly for the mixture
    upr_hrm = Mixture(n_components, mix_sub)

    return SymmetricHMoG(
        lwr_hrm,
        upr_hrm,
    )


def analytic_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    lat_dim: int,
    n_components: int,
) -> AnalyticHMoG:
    """Create an analytic hierarchical mixture of Gaussians model.

    Enables closed-form expectation-maximization for learning and bidirectional parameter
    conversion between natural and mean coordinates. Requires full covariance Gaussians in
    the latent space for complete analytical tractability.
    """

    lat_man = Normal(lat_dim, PositiveDefinite())
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)
    upr_hrm = AnalyticMixture(lat_man, n_components)

    return AnalyticHMoG(
        lwr_hrm,
        upr_hrm,
    )
