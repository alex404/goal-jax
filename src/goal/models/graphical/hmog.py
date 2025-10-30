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
from typing import Any, override

from jax import Array

from ...geometry import PositiveDefinite
from ...geometry.exponential_family.base import Mean, Natural
from ...geometry.exponential_family.graphical import (
    LatentHarmoniumEmbedding,
    ObservableEmbedding,
    hierarchical_conjugation_parameters,
    hierarchical_to_natural_likelihood,
)
from ...geometry.exponential_family.harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    SymmetricConjugated,
)
from ...geometry.manifold.base import Point
from ...geometry.manifold.embedding import LinearComposedEmbedding, LinearEmbedding
from ...geometry.manifold.linear import AffineMap
from ...geometry.manifold.matrix import Rectangular
from ..base.gaussian.generalized import Euclidean
from ..base.gaussian.normal import FullNormal, Normal
from ..harmonium.lgm import (
    GeneralizedGaussianLocationEmbedding,
    NormalAnalyticLGM,
    NormalCovarianceEmbedding,
    NormalLGM,
)
from ..harmonium.mixture import AnalyticMixture, Mixture

### HMoG Classes ###


@dataclass(frozen=True)
class DifferentiableHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    DifferentiableConjugated[
        Rectangular,  # IntRep
        Normal[ObsRep],  # Observable
        Euclidean,  # IntObservable
        Euclidean,
        AnalyticMixture[Normal[LatRep]],  # PostLatent
        Mixture[FullNormal, Normal[LatRep]],  # PriorLatent
    ]
):
    """Differentiable Hierarchical Mixture of Gaussians.

    This model combines:
    1. A linear Gaussian model (factor analysis) mapping observations to latents
    2. A Gaussian mixture model over the latent space

    Supports gradient-based optimization via log-likelihood descent.
    Uses full covariance Gaussians in the latent space.

    **Posterior vs Prior Structure**: The posterior latent mixture (`pst_upr_hrm`) uses an AnalyticMixture with a restricted covariance structure (`Normal[LatRep]`) for computational efficiency during frequent posterior computation. The prior latent mixture (`prr_upr_hrm`) is a Mixture that embeds the restricted latent covariance structure (`Normal[LatRep]`) into a full covariance structure (`FullNormal`). This embedding is necessary because the prior's shape is dictated by the conjugation parameters—it must accommodate the complete parameterization needed for conjugation parameter computation and gradient-based optimization.

    The model has three component harmoniums:
    - lwr_hrm: Maps observable Normal[ObsRep] to latent FullNormal
    - pst_upr_hrm: Analytic mixture for posterior (restricted covariance)
    - prr_upr_hrm: Differentiable mixture embedding restricted latent structure into full structure for conjugation
    """

    # Fields
    lwr_hrm: NormalLGM[ObsRep, LatRep]
    """Lower harmonium: linear Gaussian model from observations to first latents."""

    pst_upr_hrm: AnalyticMixture[Normal[LatRep]]
    """Posterior upper harmonium: analytic mixture with restricted covariance (Normal[LatRep]) for inference efficiency."""

    prr_upr_hrm: Mixture[FullNormal, Normal[LatRep]]
    """Prior upper harmonium: differentiable mixture that embeds the restricted latent covariance (Normal[LatRep]) into full covariance (FullNormal) to accommodate conjugation parameters."""

    def __post_init__(self):
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.prr_man == self.prr_upr_hrm.obs_man
        assert self.lwr_hrm.pst_man == self.pst_upr_hrm.obs_man

    # Overrides from DifferentiableConjugated

    @property
    @override
    def int_rep(self) -> Rectangular:
        return self.lwr_hrm.snd_man.rep

    @property
    @override
    def int_obs_emb(self) -> GeneralizedGaussianLocationEmbedding[Normal[ObsRep]]:
        return self.lwr_hrm.int_obs_emb

    @property
    @override
    def int_pst_emb(
        self,
    ) -> LinearEmbedding[Euclidean, AnalyticMixture[Normal[LatRep]]]:
        return LinearComposedEmbedding(
            ObservableEmbedding(self.pst_upr_hrm),
            self.lwr_hrm.int_pst_emb,
        )

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> LinearEmbedding[
        AnalyticMixture[Normal[LatRep]],
        Mixture[FullNormal, Normal[LatRep]],
    ]:
        # For DifferentiableHMoG, we need LatentHarmoniumEmbedding

        return LatentHarmoniumEmbedding(
            self.lwr_hrm.pst_prr_emb,
            self.pst_upr_hrm,
            self.prr_upr_hrm,
        )

    @override
    def extract_likelihood_input(self, z_sample: Array) -> Array:
        """Extract y (first latent) from yz sample for likelihood evaluation."""
        return z_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, FullNormal, Normal[ObsRep], Normal[ObsRep]]
        ],
    ) -> Point[Natural, Mixture[FullNormal, Normal[LatRep]]]:
        """Compute conjugation parameters for the hierarchical structure."""
        return hierarchical_conjugation_parameters(
            self.lwr_hrm, self.prr_upr_hrm, lkl_params
        )


@dataclass(frozen=True)
class SymmetricHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    SymmetricConjugated[
        Rectangular,  # IntRep
        Normal[ObsRep],  # Observable
        Normal[ObsRep],  # IntObservable
        FullNormal,  # IntLatent
        Mixture[FullNormal, Normal[LatRep]],  # PriorLatent
    ]
):
    """Symmetric Hierarchical Mixture of Gaussians.

    This model supports gradient-based optimization with additional functionality
    (e.g., join_conjugated) not available in DifferentiableHMoG.

    The symmetric structure means the posterior and conjugated latent spaces are
    the same, enabling bidirectional parameter transformations.

    Trade-off: Matrix inversions happen in the space of full covariance matrices
    over the latent space, which can be slower than DifferentiableHMoG.

    The model has two component harmoniums:
    - lwr_hrm: Maps observable Normal[ObsRep] to latent FullNormal (analytic version)
    - upr_hrm: Differentiable mixture model with constrained latent space Normal[LatRep]
    """

    # Fields
    lwr_hrm: NormalAnalyticLGM[ObsRep]
    """Lower harmonium: analytic linear Gaussian model."""

    upr_hrm: Mixture[FullNormal, Normal[LatRep]]
    """Upper harmonium: mixture model with constrained latent representation."""

    def __post_init__(self):
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.pst_man == self.upr_hrm.obs_man

    # Overrides from SymmetricConjugated

    @property
    @override
    def int_rep(self) -> Rectangular:
        return self.lwr_hrm.snd_man.rep

    @property
    @override
    def int_obs_emb(self) -> GeneralizedGaussianLocationEmbedding[Normal[ObsRep]]:
        return self.lwr_hrm.int_obs_emb

    @property
    @override
    def int_pst_emb(
        self,
    ) -> LinearEmbedding[Euclidean, Mixture[FullNormal, Normal[LatRep]]]:
        return LinearComposedEmbedding(
            ObservableEmbedding(self.upr_hrm),
            self.lwr_hrm.int_pst_emb,
        )

    @property
    @override
    def lat_man(self) -> Mixture[FullNormal, Normal[LatRep]]:
        return self.upr_hrm

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract y (first latent) from yz sample for likelihood evaluation."""
        return prr_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, FullNormal, Normal[ObsRep], Normal[ObsRep]]
        ],
    ) -> Point[Natural, Mixture[FullNormal, Normal[LatRep]]]:
        """Compute conjugation parameters for the hierarchical structure."""
        return hierarchical_conjugation_parameters(
            self.lwr_hrm, self.upr_hrm, lkl_params
        )


@dataclass(frozen=True)
class AnalyticHMoG[ObsRep: PositiveDefinite](  # pyright: ignore[reportGeneralTypeIssues]
    AnalyticConjugated[
        Rectangular,  # IntRep
        Normal[ObsRep],  # Observable
        Normal[ObsRep],  # IntObservable
        FullNormal,  # IntLatent
        AnalyticMixture[FullNormal],  # PriorLatent
    ],
    SymmetricHMoG[ObsRep, PositiveDefinite],
):
    """Analytic Hierarchical Mixture of Gaussians.

    This model enables:
    - Closed-form EM algorithm for learning (from AnalyticConjugated)
    - Bidirectional parameter conversion (mean ↔ natural)
    - Full analytical tractability
    - Shared implementation with SymmetricHMoG

    Requires full covariance Gaussians in the latent space.

    The model uses:
    - lwr_hrm: Analytic linear Gaussian model
    - upr_hrm: Analytic mixture model (all operations have closed forms)
    """

    # Override upr_hrm type to be AnalyticMixture (stricter than parent's Mixture)
    upr_hrm: AnalyticMixture[FullNormal]
    """Upper harmonium: analytic mixture model."""

    @override
    def to_natural_likelihood(
        self,
        params: Point[Mean, Any],
    ) -> Point[
        Natural, AffineMap[Rectangular, FullNormal, Normal[ObsRep], Normal[ObsRep]]
    ]:
        """Convert mean parameters to natural likelihood parameters."""
        return hierarchical_to_natural_likelihood(
            self, self.lwr_hrm, self.upr_hrm, params
        )


## Factory Functions ##


def differentiable_hmog[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    pst_lat_rep: type[LatRep],
    n_components: int,
) -> DifferentiableHMoG[ObsRep, LatRep]:
    """Create a differentiable hierarchical mixture of Gaussians model.

    This function constructs a hierarchical model combining:
    1. A bottom layer with a linear Gaussian model reducing observables to first-level latents
    2. A top layer with a Gaussian mixture model for modelling the latent distribution

    This model supports optimization via log-likelihood gradient descent.
    Uses full covariance Gaussians in the latent space.
    """
    pst_y_man = Normal(lat_dim, pst_lat_rep)
    prr_y_man = Normal(lat_dim, PositiveDefinite)
    lwr_hrm = NormalLGM(obs_dim, obs_rep, lat_dim, pst_lat_rep)
    mix_sub = NormalCovarianceEmbedding(pst_y_man, prr_y_man)
    pst_upr_hrm = AnalyticMixture(pst_y_man, n_components)
    prr_upr_hrm = Mixture(n_components, mix_sub)

    return DifferentiableHMoG(
        lwr_hrm,
        pst_upr_hrm,
        prr_upr_hrm,
    )


def symmetric_hmog[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    lat_rep: type[LatRep],
    n_components: int,
) -> SymmetricHMoG[ObsRep, LatRep]:
    """Create a symmetric hierarchical mixture of Gaussians model.

    Supports optimization via log-likelihood gradient descent with additional functionality
    (e.g., `join_conjugated`) not available in `DifferentiableHMoG`. The symmetric structure
    means posterior and prior use the same latent parameterization.

    Trade-off: Matrix inversions happen in the space of full covariance matrices over the
    latent space, which can be slower than `DifferentiableHMoG`.
    """
    mid_lat_man = Normal(lat_dim, PositiveDefinite)
    sub_lat_man = Normal(lat_dim, lat_rep)
    mix_sub = NormalCovarianceEmbedding(sub_lat_man, mid_lat_man)
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)
    upr_hrm = Mixture(n_components, mix_sub)

    return SymmetricHMoG(
        lwr_hrm,
        upr_hrm,
    )


def analytic_hmog[ObsRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    n_components: int,
) -> AnalyticHMoG[ObsRep]:
    """Create an analytic hierarchical mixture of Gaussians model.

    Enables closed-form expectation-maximization for learning and bidirectional parameter
    conversion between natural and mean coordinates. Requires full covariance Gaussians in
    the latent space for complete analytical tractability.
    """

    lat_man = Normal(lat_dim, PositiveDefinite)
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)
    upr_hrm = AnalyticMixture(lat_man, n_components)

    return AnalyticHMoG(
        lwr_hrm,
        upr_hrm,
    )
