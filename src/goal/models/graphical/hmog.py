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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticConjugated,
    DifferentiableConjugated,
    LinearEmbedding,
    ObservableEmbedding,
    PositiveDefinite,
    RootEmbedding,
    SymmetricConjugated,
)
from ..base.gaussian.normal import FullNormal, Normal, full_normal
from ..harmonium.lgm import (
    NormalAnalyticLGM,
    NormalCovarianceEmbedding,
    NormalLGM,
)
from ..harmonium.mixture import AnalyticMixture, CompleteMixture, Mixture

# HMoG Classes


class _HMoGBase[
    LowerHarmonium: DifferentiableConjugated[Any, Any, Any],
    PstUpperHarmonium: CompleteMixture[Any],
    PrrUpperHarmonium: DifferentiableConjugated[Any, Any, Any],
](
    DifferentiableConjugated[Any, PstUpperHarmonium, PrrUpperHarmonium],
    ABC,
):
    """Abstract base for Hierarchical Mixture of Gaussians models.

    Composes a lower harmonium ($x \\to y$) with an upper mixture ($y \\to k$) over the
    three-node chain. The lower harmonium supplies the root and cross spans, the upper
    mixture is the deep span, and the upper mixture's own root span is node $y$ --- which
    is why the two compose without any coordinate translation.

    The ``pst_upr_hrm`` is bounded by ``CompleteMixture``, giving access to
    ``split_mean_mixture``, ``join_mean_mixture``, and ``cmp_man``.
    """

    # Contract

    @property
    @abstractmethod
    def lwr_hrm(self) -> LowerHarmonium:
        """Lower harmonium (observable to middle latent)."""

    @property
    @abstractmethod
    def pst_upr_hrm(self) -> PstUpperHarmonium:
        """Posterior upper harmonium (possibly restricted)."""

    @property
    @abstractmethod
    def prr_upr_hrm(self) -> PrrUpperHarmonium:
        """Prior upper harmonium (for conjugation)."""

    # Overrides

    @property
    @override
    def int_man(self) -> Any:
        """The lower harmonium's interaction, re-aimed at node $y$ inside the upper mixture."""
        return self.lwr_hrm.int_man.prepend_embedding(
            ObservableEmbedding(self.pst_upr_hrm)
        )

    @property
    @override
    def pst_prr_emb(self) -> LinearEmbedding[PstUpperHarmonium, PrrUpperHarmonium]:
        """The posterior and prior mixtures differ only at node $y$, i.e. in their root span."""
        return RootEmbedding(
            self.lwr_hrm.pst_prr_emb,
            self.pst_upr_hrm,
            self.prr_upr_hrm,
        )

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        return prr_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Place the lower harmonium's conjugation parameters into node $y$'s slot of the upper mixture."""
        return ObservableEmbedding(self.prr_upr_hrm).embed(
            self.lwr_hrm.conjugation_parameters(lkl_params)
        )

    # Methods

    def whiten_prior(self, means: Array) -> Array:
        """Reparameterize the latent Y-space to have zero mean and identity covariance.

        Preserves p(x) by updating both:
        - The lower LGM interaction (loading matrix + observable bias adjustment)
        - Each GMM component (via the existing Normal.whiten relative to GMM marginal)
        """
        obs_means, lwr_int_means, lat_means = self.split_level(means)

        # GMM marginal statistics (obs_means_gmm = E[s_Y(y)] w.r.t. joint)
        obs_means_gmm, _, cat_means = self.pst_upr_hrm.split_level(lat_means)
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
        lwr_int_mat = self.lwr_hrm.int_man.to_matrix(lwr_int_means)  # pyright: ignore[reportAttributeAccessIssue]
        cross_cov = lwr_int_mat - jnp.outer(obs_loc, lat_mean_y)  # W Cov(Y)
        new_lwr_int_mat = jax.scipy.linalg.solve_triangular(
            chol, cross_cov.T, lower=True
        ).T
        new_lwr_int_means = self.lwr_hrm.int_man.from_matrix(new_lwr_int_mat)  # pyright: ignore[reportAttributeAccessIssue]

        return self.join_level(obs_means, new_lwr_int_means, new_lat_means)

    def posterior_categorical(self, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates."""
        return self.pst_upr_hrm.prior(self.posterior_at(params, x))

    def posterior_soft_assignments(self, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x)."""
        cat_natural = self.posterior_categorical(params, x)
        cat_means = self.pst_upr_hrm.lat_man.to_mean(cat_natural)
        return self.pst_upr_hrm.lat_man.to_probs(cat_means)

    def posterior_hard_assignment(self, params: Array, x: Array) -> Array:
        """Compute hard assignment to most probable component."""
        return jnp.argmax(self.posterior_soft_assignments(params, x))


@dataclass(frozen=True)
class DifferentiableHMoG[ObsRep: PositiveDefinite, PstRep: PositiveDefinite](
    _HMoGBase[
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
    SymmetricConjugated[Any, Upr],
    _HMoGBase[NormalAnalyticLGM[ObsRep], Upr, Upr],
    ABC,
):
    """Symmetric HMoG base class.

    The symmetric structure means ``pst_upr_hrm = prr_upr_hrm = upr_hrm``,
    enabling bidirectional parameter transformations like ``join_conjugated``.

    Trade-off: Matrix inversions happen in the space of full covariance matrices
    over the latent space, which can be slower than DifferentiableHMoG.
    """

    # Contract

    @property
    @abstractmethod
    def upr_hrm(self) -> Upr:
        """Upper harmonium (middle latent to top latent)."""

    # Overrides

    @property
    @override
    def lat_man(self) -> Upr:
        return self.upr_hrm

    @property
    @override
    def pst_upr_hrm(self) -> Upr:
        return self.upr_hrm

    @property
    @override
    def prr_upr_hrm(self) -> Upr:
        return self.upr_hrm


@dataclass(frozen=True)
class AnalyticHMoG[ObsRep: PositiveDefinite](
    SymmetricHMoG[ObsRep, AnalyticMixture[FullNormal]],
    AnalyticConjugated[Any, AnalyticMixture[FullNormal]],
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
    def to_natural_likelihood(self, means: Array) -> Array:
        """Project the mean parameters down onto the lower harmonium and convert there.

        The deep span is the upper mixture; its own root span is node $y$, which is the
        lower harmonium's latent side. So the projection is one nested root read.
        """
        obs_means, lwr_int_means, lat_means = self.split_level(means)
        lwr_lat_means = ObservableEmbedding(self.upr_hrm).project(lat_means)
        lwr_means = self.lwr_hrm.join_level(obs_means, lwr_int_means, lwr_lat_means)
        return self.lwr_hrm.to_natural_likelihood(lwr_means)

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
