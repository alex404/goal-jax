"""Hierarchical Mixture of Gaussians (HMoG) models.

This module provides concrete implementations of hierarchical Gaussian models,
avoiding the complexity of fully generic hierarchical harmoniums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from jax import Array

from ...geometry import IdentityEmbedding, PositiveDefinite
from ...geometry.exponential_family.base import Mean, Natural
from ...geometry.exponential_family.graphical import (
    LatentHarmoniumEmbedding,
    ObservableEmbedding,
    hierarchical_conjugation_parameters,  # pyright: ignore[reportUnknownVariableType]
    hierarchical_sample,  # pyright: ignore[reportUnknownVariableType]
    hierarchical_to_natural_likelihood,  # pyright: ignore[reportUnknownVariableType]
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
from ..base.gaussian.normal import FullNormal, Normal
from ..harmonium.lgm import (
    NormalAnalyticLinearGaussianModel,
    NormalCovarianceEmbedding,
    NormalDifferentiableLinearGaussianModel,
)
from ..harmonium.mixture import AnalyticMixture, DifferentiableMixture

### HMoG Classes ###


@dataclass(frozen=True)
class DifferentiableHMoG[ObsRep: PositiveDefinite](
    DifferentiableConjugated[
        Rectangular,  # IntRep
        Normal[ObsRep],  # Observable
        Normal[ObsRep],  # IntObservable
        FullNormal,  # IntLatent
        AnalyticMixture[FullNormal],  # PostLatent
        DifferentiableMixture[FullNormal, FullNormal],  # Latent
    ]
):
    """Differentiable Hierarchical Mixture of Gaussians.

    This model combines:
    1. A linear Gaussian model (factor analysis) mapping observations to latents
    2. A Gaussian mixture model over the latent space

    Supports gradient-based optimization via log-likelihood descent.
    Uses full covariance Gaussians in the latent space.

    The model has three component harmoniums:
    - lwr_hrm: Maps observable Normal[ObsRep] to latent FullNormal
    - upr_hrm: Analytic mixture model (for posterior computation)
    - con_upr_hrm: Differentiable mixture model (for conjugation/optimization)
    """

    # Fields
    lwr_hrm: NormalDifferentiableLinearGaussianModel[ObsRep]
    """Lower harmonium: linear Gaussian model from observations to first latents."""

    upr_hrm: AnalyticMixture[FullNormal]
    """Upper harmonium: analytic mixture for posterior."""

    con_upr_hrm: DifferentiableMixture[FullNormal, FullNormal]
    """Conjugated upper harmonium: differentiable mixture for optimization."""

    def __post_init__(self):
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.con_lat_man == self.con_upr_hrm.obs_man
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man

    # Overrides from DifferentiableConjugated

    @property
    @override
    def int_rep(self) -> Rectangular:
        return self.lwr_hrm.snd_man.rep

    @property
    @override
    def obs_emb(self) -> LinearEmbedding[Normal[ObsRep], Normal[ObsRep]]:
        return self.lwr_hrm.obs_emb  # pyright: ignore[reportReturnType]

    @property
    @override
    def int_lat_emb(
        self,
    ) -> LinearEmbedding[FullNormal, DifferentiableMixture[FullNormal, FullNormal]]:
        return LinearComposedEmbedding(  # pyright: ignore[reportReturnType]
            ObservableEmbedding(self.upr_hrm),
            self.lwr_hrm.int_lat_emb,
        )

    @property
    @override
    def pst_lat_emb(
        self,
    ) -> LinearEmbedding[
        AnalyticMixture[FullNormal], DifferentiableMixture[FullNormal, FullNormal]
    ]:
        # For DifferentiableHMoG, we need LatentHarmoniumEmbedding

        return LatentHarmoniumEmbedding(  # pyright: ignore[reportReturnType]
            self.lwr_hrm.pst_lat_emb,
            self.upr_hrm,
            self.con_upr_hrm,
        )

    @override
    def sample(self, key: Array, params: Point[Natural, Any], n: int = 1) -> Array:
        """Sample from the hierarchical model using ancestral sampling."""
        return hierarchical_sample(self, self.lwr_hrm, self.con_lat_man, key, params, n)

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, FullNormal, Normal[ObsRep], Normal[ObsRep]]
        ],
    ) -> Point[Natural, DifferentiableMixture[FullNormal, FullNormal]]:
        """Compute conjugation parameters for the hierarchical structure."""
        return hierarchical_conjugation_parameters(
            self.lwr_hrm, self.con_upr_hrm, lkl_params
        )


@dataclass(frozen=True)
class SymmetricHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    SymmetricConjugated[
        Rectangular,  # IntRep
        Normal[ObsRep],  # Observable
        Normal[ObsRep],  # IntObservable
        FullNormal,  # IntLatent
        DifferentiableMixture[FullNormal, Normal[LatRep]],  # Latent
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
    lwr_hrm: NormalAnalyticLinearGaussianModel[ObsRep]
    """Lower harmonium: analytic linear Gaussian model."""

    upr_hrm: DifferentiableMixture[FullNormal, Normal[LatRep]]
    """Upper harmonium: mixture model with constrained latent representation."""

    def __post_init__(self):
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man

    # Overrides from SymmetricConjugated

    @property
    @override
    def int_rep(self) -> Rectangular:
        return self.lwr_hrm.snd_man.rep

    @property
    @override
    def obs_emb(self) -> LinearEmbedding[Normal[ObsRep], Normal[ObsRep]]:
        return self.lwr_hrm.obs_emb  # pyright: ignore[reportReturnType]

    @property
    @override
    def int_lat_emb(
        self,
    ) -> LinearEmbedding[FullNormal, DifferentiableMixture[FullNormal, Normal[LatRep]]]:
        return LinearComposedEmbedding(  # pyright: ignore[reportReturnType]
            ObservableEmbedding(self.upr_hrm),
            self.lwr_hrm.int_lat_emb,
        )

    @property
    @override
    def pst_lat_emb(
        self,
    ) -> LinearEmbedding[
        DifferentiableMixture[FullNormal, Normal[LatRep]],
        DifferentiableMixture[FullNormal, Normal[LatRep]],
    ]:
        # Symmetric: posterior and conjugated are the same, so identity embedding

        return IdentityEmbedding(self.upr_hrm)

    @override
    def sample(self, key: Array, params: Point[Natural, Any], n: int = 1) -> Array:
        """Sample from the hierarchical model using ancestral sampling."""
        return hierarchical_sample(self, self.lwr_hrm, self.con_lat_man, key, params, n)

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, FullNormal, Normal[ObsRep], Normal[ObsRep]]
        ],
    ) -> Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]]:
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
        AnalyticMixture[FullNormal],  # Latent
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

    # Override upr_hrm type to be AnalyticMixture (stricter than parent's DifferentiableMixture)
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


def differentiable_hmog[ObsRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    n_components: int,
) -> DifferentiableHMoG[ObsRep]:
    """Create a differentiable hierarchical mixture of Gaussians model.

    This function constructs a hierarchical model combining:
    1. A bottom layer with a linear Gaussian model reducing observables to first-level latents
    2. A top layer with a Gaussian mixture model for modelling the latent distribution

    This model supports optimization via log-likelihood gradient descent.
    Uses full covariance Gaussians in the latent space.
    """
    lat_man = Normal(lat_dim, PositiveDefinite)
    lwr_hrm = NormalDifferentiableLinearGaussianModel(obs_dim, obs_rep, lat_dim)
    upr_hrm = AnalyticMixture(lat_man, n_components)
    con_upr_hrm = DifferentiableMixture(n_components, IdentityEmbedding(lat_man))

    return DifferentiableHMoG(
        lwr_hrm,
        upr_hrm,
        con_upr_hrm,
    )


def symmetric_hmog[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    lat_rep: type[LatRep],
    n_components: int,
) -> SymmetricHMoG[ObsRep, LatRep]:
    """Create a symmetric hierarchical mixture of Gaussians model. Supports optimization via log-likelihood gradient descent, but can be slower since requisite matrix inversions happen in the space of full covariance matrices over the latent space. Nevertheless, the supports additional functionality (e.g. join_conjugated) not available to `differentiableHMoG`."""
    mid_lat_man = Normal(lat_dim, PositiveDefinite)
    sub_lat_man = Normal(lat_dim, lat_rep)
    mix_sub = NormalCovarianceEmbedding(sub_lat_man, mid_lat_man)
    lwr_hrm = NormalAnalyticLinearGaussianModel(obs_dim, obs_rep, lat_dim)
    upr_hrm = DifferentiableMixture(n_components, mix_sub)

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
    """Create an analytic hierarchical mixture of Gaussians model. The resulting model enables closed-form EM for learning and bidirectional parameter conversion, however requires mixtuers of full coviarance Gaussians in the latent space."""

    lat_man = Normal(lat_dim, PositiveDefinite)
    lwr_hrm = NormalAnalyticLinearGaussianModel(obs_dim, obs_rep, lat_dim)
    upr_hrm = AnalyticMixture(lat_man, n_components)

    return AnalyticHMoG(
        lwr_hrm,
        upr_hrm,
    )
