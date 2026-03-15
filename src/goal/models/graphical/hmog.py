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

**Variants**:

- **DifferentiableHMoG**: Gradient-based optimization, uses restricted posterior covariance
  for efficiency (e.g., diagonal)
- **SymmetricHMoG**: Abstract base for symmetric HMoG variants, provides shared posterior methods
- **AnalyticHMoG**: Fully analytic, enables closed-form EM and bidirectional parameter conversion

Factory functions (``differentiable_hmog``, ``analytic_hmog``) provide convenient construction
for common configurations.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, override

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
from ..harmonium.mixture import AnalyticMixture, CompleteMixture, Mixture

# HMoG Classes


class HMoGMethods(ABC):
    """Mixin providing shared HMoG posterior and whitening methods.

    Requires ``pst_upr_hrm`` (a ``CompleteMixture``), ``lwr_hrm``, ``obs_man``,
    ``split_coords``, ``join_coords``, and ``posterior_at`` — all available on
    both ``DifferentiableHierarchical`` and ``SymmetricHierarchical``.
    """

    def whiten_prior(self: Any, means: Array) -> Array:
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
        cross_cov = lwr_int_mat - jnp.outer(obs_loc, lat_mean_y)  # W Cov(Y)
        new_lwr_int_mat = jax.scipy.linalg.solve_triangular(
            chol, cross_cov.T, lower=True
        ).T
        new_lwr_int_means = self.lwr_hrm.int_man.from_matrix(new_lwr_int_mat)

        return self.join_coords(obs_means, new_lwr_int_means, new_lat_means)

    def posterior_categorical(self: Any, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates."""
        return self.pst_upr_hrm.prior(self.posterior_at(params, x))

    def posterior_soft_assignments(self: Any, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x)."""
        cat_natural = self.posterior_categorical(params, x)
        cat_means = self.pst_upr_hrm.lat_man.to_mean(cat_natural)
        return self.pst_upr_hrm.lat_man.to_probs(cat_means)

    def posterior_hard_assignment(self: Any, params: Array, x: Array) -> Array:
        """Compute hard assignment to most probable component."""
        return jnp.argmax(self.posterior_soft_assignments(params, x))


@dataclass(frozen=True)
class DifferentiableHMoG[ObsRep: PositiveDefinite, PstRep: PositiveDefinite](
    HMoGMethods,
    DifferentiableHierarchical[
        NormalLGM[ObsRep, PstRep],
        AnalyticMixture[Normal[PstRep]],
        Mixture[FullNormal],
    ],
):
    """Differentiable Hierarchical Mixture of Gaussians.

    Combines a linear Gaussian model (factor analysis) mapping observations to latents
    with a Gaussian mixture model over the latent space. Supports gradient-based
    optimization via log-likelihood descent.

    **Posterior vs Prior Structure**: The posterior latent mixture (``pst_upr_hrm``) uses an
    AnalyticMixture with a restricted covariance structure for computational efficiency.
    The prior latent mixture (``prr_upr_hrm``) embeds the restricted structure into full
    covariance for conjugation parameter computation.
    """

    # Fields

    _lwr_hrm: NormalLGM[ObsRep, PstRep]
    _pst_upr_hrm: AnalyticMixture[Normal[PstRep]]
    _prr_upr_hrm: Mixture[FullNormal]

    # Overrides

    @property
    @override
    def lwr_hrm(self) -> NormalLGM[ObsRep, PstRep]:
        return self._lwr_hrm

    @property
    @override
    def pst_upr_hrm(self) -> AnalyticMixture[Normal[PstRep]]:
        return self._pst_upr_hrm

    @property
    @override
    def prr_upr_hrm(self) -> Mixture[FullNormal]:
        return self._prr_upr_hrm


class SymmetricHMoG[ObsRep: PositiveDefinite, Upr: CompleteMixture[Any]](
    HMoGMethods,
    SymmetricHierarchical[NormalAnalyticLGM[ObsRep], Upr],
    ABC,
):
    """Symmetric HMoG base class.

    The symmetric structure means ``pst_upr_hrm = prr_upr_hrm = upr_hrm``,
    enabling bidirectional parameter transformations like ``join_conjugated``.

    Trade-off: Matrix inversions happen in the space of full covariance matrices
    over the latent space, which can be slower than DifferentiableHMoG.
    """


@dataclass(frozen=True)
class AnalyticHMoG[ObsRep: PositiveDefinite](
    SymmetricHMoG[ObsRep, AnalyticMixture[FullNormal]],
    AnalyticHierarchical[NormalAnalyticLGM[ObsRep], AnalyticMixture[FullNormal]],
):
    """Analytic Hierarchical Mixture of Gaussians.

    Enables closed-form EM and bidirectional parameter conversion (mean <-> natural).
    Requires full covariance Gaussians in the latent space.
    """

    # Fields

    _lwr_hrm: NormalAnalyticLGM[ObsRep]
    _upr_hrm: AnalyticMixture[FullNormal]

    # Overrides

    @property
    @override
    def lwr_hrm(self) -> NormalAnalyticLGM[ObsRep]:
        return self._lwr_hrm

    @property
    @override
    def upr_hrm(self) -> AnalyticMixture[FullNormal]:
        return self._upr_hrm

    @override
    def expectation_maximization(self, params: Array, xs: Array) -> Array:
        """Perform a single iteration of EM with latent-prior whitening.

        HMoG has the same latent-space non-identifiability as FA/PCA. After the
        E-step, whiten the latent prior in mean coordinates before mapping back
        to natural coordinates.
        """
        q = self.mean_posterior_statistics(params, xs)
        return self.to_natural(self.whiten_prior(q))


# Factory Functions


def differentiable_hmog[ObsRep: PositiveDefinite, PstRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: ObsRep,
    lat_dim: int,
    pst_rep: PstRep,
    n_components: int,
) -> DifferentiableHMoG[ObsRep, PstRep]:
    """Create a differentiable hierarchical mixture of Gaussians model.

    Combines a linear Gaussian model reducing observables to first-level latents
    with a Gaussian mixture model over the latent distribution. Supports optimization
    via log-likelihood gradient descent.
    """
    pst_y_man = Normal(lat_dim, pst_rep)
    prr_y_man = full_normal(lat_dim)
    lwr_hrm = NormalLGM(obs_dim, obs_rep, lat_dim, pst_rep)
    mix_sub = NormalCovarianceEmbedding(pst_y_man, prr_y_man)
    pst_upr_hrm = AnalyticMixture(pst_y_man, n_components)

    prr_upr_hrm = Mixture(n_components, mix_sub)

    return DifferentiableHMoG(
        _lwr_hrm=lwr_hrm,
        _pst_upr_hrm=pst_upr_hrm,
        _prr_upr_hrm=prr_upr_hrm,
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
        _lwr_hrm=lwr_hrm,
        _upr_hrm=upr_hrm,
    )
