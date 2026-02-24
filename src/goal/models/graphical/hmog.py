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

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticHierarchical,
    DifferentiableHierarchical,
    PositiveDefinite,
    SymmetricHierarchical,
)
from ..base.gaussian.normal import FullNormal, Normal, full_normal
from ..harmonium.lgm import (
    NormalAnalyticLGM,
    NormalCovarianceEmbedding,
    NormalLGM,
)
from ..harmonium.mixture import AnalyticMixture, Mixture

### HMoG Classes ###


# TODO: Somehow figure out how to share more code between these classes while preserving the type system.
@dataclass(frozen=True)
class DifferentiableHMoG[ObsRep: PositiveDefinite, PstRep: PositiveDefinite](
    DifferentiableHierarchical[
        NormalLGM[ObsRep, PstRep],
        AnalyticMixture[Normal[PstRep]],
        Mixture[FullNormal],
    ]
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

    def whiten_prior(self, means: Array) -> Array:
        """Reparameterize the latent Y-space to have zero mean and identity covariance.

        Preserves p(x) by updating both:
        - The lower LGM interaction (loading matrix + observable bias adjustment)
        - Each GMM component (via the existing Normal.whiten relative to GMM marginal)
        """
        obs_means, lwr_int_means, lat_means = self.split_coords(means)

        # GMM marginal statistics (obs_means_gmm = E[s_Y(y)] w.r.t. joint)
        obs_means_gmm, _, cat_means = self.pst_upr_hrm.split_coords(lat_means)
        lat_mean_y, lat_cov_y = self.pst_upr_hrm.obs_man.split_mean_covariance(
            obs_means_gmm
        )
        chol = jnp.linalg.cholesky(
            self.pst_upr_hrm.obs_man.cov_man.to_matrix(lat_cov_y)
        )

        # Whiten each GMM component using existing Normal.whiten
        comp_means, _ = self.pst_upr_hrm.split_mean_mixture(lat_means)
        new_comp_means = self.pst_upr_hrm.cmp_man.map(
            lambda c: self.pst_upr_hrm.obs_man.relative_whiten(c, obs_means_gmm),
            comp_means,
            flatten=True,
        )
        new_lat_means = self.pst_upr_hrm.join_mean_mixture(new_comp_means, cat_means)

        # Update lower LGM cross-statistics (same transform as LGM whitening)
        obs_loc, _ = self.obs_man.split_mean_second_moment(obs_means)
        lwr_int_mat = self.lwr_hrm.int_man.to_matrix(lwr_int_means)
        cross_cov = lwr_int_mat - jnp.outer(obs_loc, lat_mean_y)  # W·Cov(Y)
        new_lwr_int_mat = jax.scipy.linalg.solve_triangular(
            chol, cross_cov.T, lower=True
        ).T
        new_lwr_int_means = self.lwr_hrm.int_man.from_matrix(new_lwr_int_mat)

        return self.join_coords(obs_means, new_lwr_int_means, new_lat_means)

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
class SymmetricHMoG[ObsRep: PositiveDefinite](
    SymmetricHierarchical[NormalAnalyticLGM[ObsRep], Mixture[FullNormal]]
):
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
class AnalyticHMoG[ObsRep: PositiveDefinite](
    AnalyticHierarchical[NormalAnalyticLGM[ObsRep], AnalyticMixture[FullNormal]]
):
    """Analytic Hierarchical Mixture of Gaussians.

    This model enables:
    - Closed-form EM algorithm for learning (from AnalyticConjugated)
    - Bidirectional parameter conversion (mean ↔ natural)
    - Full analytical tractability

    Requires full covariance Gaussians in the latent space.
    """

    def whiten_prior(self, means: Array) -> Array:
        """Reparameterize the latent Y-space to have zero mean and identity covariance.

        Preserves p(x) by updating both:
        - The lower LGM interaction (loading matrix + observable bias adjustment)
        - Each GMM component (via the existing Normal.whiten relative to GMM marginal)
        """
        obs_means, lwr_int_means, lat_means = self.split_coords(means)

        # GMM marginal statistics (obs_means_gmm = E[s_Y(y)] w.r.t. joint)
        obs_means_gmm, _, cat_means = self.upr_hrm.split_coords(lat_means)
        lat_mean_y, lat_cov_y = self.upr_hrm.obs_man.split_mean_covariance(
            obs_means_gmm
        )
        chol = jnp.linalg.cholesky(self.upr_hrm.obs_man.cov_man.to_matrix(lat_cov_y))

        # Whiten each GMM component using existing Normal.whiten
        comp_means, _ = self.upr_hrm.split_mean_mixture(lat_means)
        new_comp_means = self.upr_hrm.cmp_man.map(
            lambda c: self.upr_hrm.obs_man.relative_whiten(c, obs_means_gmm),
            comp_means,
            flatten=True,
        )
        new_lat_means = self.upr_hrm.join_mean_mixture(new_comp_means, cat_means)

        # Update lower LGM cross-statistics (same transform as LGM whitening)
        obs_loc, _ = self.obs_man.split_mean_second_moment(obs_means)
        lwr_int_mat = self.lwr_hrm.int_man.to_matrix(lwr_int_means)
        cross_cov = lwr_int_mat - jnp.outer(obs_loc, lat_mean_y)  # W·Cov(Y)
        new_lwr_int_mat = jax.scipy.linalg.solve_triangular(
            chol, cross_cov.T, lower=True
        ).T
        new_lwr_int_means = self.lwr_hrm.int_man.from_matrix(new_lwr_int_mat)

        return self.join_coords(obs_means, new_lwr_int_means, new_lat_means)

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


def differentiable_hmog[ObsRep: PositiveDefinite, PstRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: ObsRep,
    lat_dim: int,
    pst_rep: PstRep,
    n_components: int,
) -> DifferentiableHMoG[ObsRep, PstRep]:
    """Create a differentiable hierarchical mixture of Gaussians model.

    This function constructs a hierarchical model combining:
    1. A bottom layer with a linear Gaussian model reducing observables to first-level latents
    2. A top layer with a Gaussian mixture model for modelling the latent distribution

    This model supports optimization via log-likelihood gradient descent.
    Uses full covariance Gaussians in the latent space.
    """

    pst_y_man = Normal(lat_dim, pst_rep)
    prr_y_man = full_normal(lat_dim)
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


def symmetric_hmog[ObsRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: ObsRep,
    lat_dim: int,
    lat_rep: PositiveDefinite,
    n_components: int,
) -> SymmetricHMoG[ObsRep]:
    """Create a symmetric hierarchical mixture of Gaussians model.

    Supports optimization via log-likelihood gradient descent with additional functionality
    (e.g., `join_conjugated`) not available in `DifferentiableHMoG`. The symmetric structure
    means posterior and prior use the same latent parameterization.

    Trade-off: Matrix inversions happen in the space of full covariance matrices over the
    latent space, which can be slower than `DifferentiableHMoG`.
    """

    mid_lat_man = full_normal(lat_dim)
    sub_lat_man = Normal(lat_dim, lat_rep)
    mix_sub = NormalCovarianceEmbedding(sub_lat_man, mid_lat_man)
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)

    # Use the embedding directly for the mixture
    upr_hrm = Mixture(n_components, mix_sub)

    return SymmetricHMoG(
        lwr_hrm,
        upr_hrm,
    )


def analytic_hmog[ObsRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: ObsRep,
    lat_dim: int,
    n_components: int,
) -> AnalyticHMoG[ObsRep]:
    """Create an analytic hierarchical mixture of Gaussians model.

    Enables closed-form expectation-maximization for learning and bidirectional parameter
    conversion between natural and mean coordinates. Requires full covariance Gaussians in
    the latent space for complete analytical tractability.
    """

    lat_man = full_normal(lat_dim)
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)
    upr_hrm = AnalyticMixture(lat_man, n_components)

    return AnalyticHMoG(
        lwr_hrm,
        upr_hrm,
    )
