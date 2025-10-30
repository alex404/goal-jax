"""Mixture of Factor Analyzers (MFA) model implemented as a conjugated harmonium.

MFA extends mixture modeling to handle latent variable structures, representing observations
as arising from a mixture of factor analysis models. Each mixture component has its own
factor analyzer with a distinct mean, but components share a common loading matrix and
noise covariance structure.

**Model comparison with related models**:

- **vs Mixture of Gaussians**: MFA adds latent factor structure (dimensionality reduction)
- **vs Factor Analysis**: MFA adds mixture component structure (clustering)
- **vs HMoG**: Different conditional relationships:
    - HMoG: :math:`p(X | Y) \\cdot p(Y | Z) \\cdot p(Z)` (hierarchical: factors → mixture)
    - MFA: Mixture and factors jointly determine observables (factors differ per component mean)

This structure is useful for:
- Heterogeneous data clusters with different means but similar underlying structure
- Simultaneous dimensionality reduction and clustering
- Modeling complex multivariate distributions with multiple modes

**Status**: The MFA module provides the core harmonium structure and fully documented class.
Some methods remain unimplemented (e.g., ``conjugation_parameters``), but the framework is
in place for complete implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from ...geometry import (
    LinearEmbedding,
    PositiveDefinite,
    Rectangular,
    SymmetricConjugated,
)
from ...geometry.exponential_family.graphical import ObservableEmbedding
from ...geometry.manifold.embedding import LinearComposedEmbedding
from ..base.gaussian.generalized import Euclidean
from ..base.gaussian.normal import FullNormal, Normal
from ..harmonium.lgm import (
    GeneralizedGaussianLocationEmbedding,
    NormalAnalyticLGM,
)
from ..harmonium.mixture import (
    AnalyticMixture,
)


@dataclass(frozen=True)
class MixtureOfLGMs[
    ObsRep: PositiveDefinite,
](
    SymmetricConjugated[
        Rectangular,
        Normal[ObsRep],
        Euclidean,
        Euclidean,
        AnalyticMixture[FullNormal],
    ],
):
    """Mixture of Factor Analyzers (MFA) as a conjugated harmonium.

    A Mixture of Factor Analyzers combines factor analysis with mixture modeling, enabling
    representation learning in heterogeneous data with multiple modes. The model assumes each
    observation is generated from one of :math:`K` factor analyzers.

    **Model specification**:

    For observations :math:`x \\in \\mathbb{R}^p`, latent factors :math:`y \\in \\mathbb{R}^d`,
    and mixture assignment :math:`z \\in \\{1,\\ldots,K\\}`:

    .. math::

        p(x, y, z) = p(z) \\cdot p(y | z) \\cdot p(x | y, z)

    where:

    - :math:`p(z) = \\text{Cat}(z; \\pi)` is the categorical mixing distribution
    - :math:`p(y | z) = \\mathcal{N}(y; \\mu_z, \\Sigma_z)` is the component-specific latent prior
    - :math:`p(x | y, z) = \\mathcal{N}(x; A_z y + b_z, \\Sigma_x)` is the factor analysis likelihood

    **Key structure**:

    - Observable variables :math:`X \\in \\mathbb{R}^p` follow a multivariate Gaussian with covariance
      parameterized by `ObsRep` (Diagonal, Isotropic, or Full)
    - Latent factors :math:`Y \\in \\mathbb{R}^d` are jointly modeled as a mixture of Gaussians
    - Component-specific parameters (means) are encoded in the interaction matrix
    - Loading matrix and noise covariance are shared across components

    **Harmonium formulation**:

    The MFA is implemented as a symmetric conjugated harmonium with:

    - Observable: :math:`\\text{Normal}[\\text{ObsRep}]` (observable distribution family)
    - Interaction matrix: Encodes component-specific offsets in factor space
    - Latent: :math:`\\text{AnalyticMixture}[\\text{FullNormal}]` (mixture of Gaussians over factors and components)

    **Posterior vs Prior Structure**:

    The posterior latent distribution uses an `AnalyticMixture` with full covariance for
    closed-form inference. The prior uses the same parameterization in the symmetric structure,
    enabling bidirectional parameter transformations and the `join_conjugated` method for
    constructing models from marginal likelihood and prior parameters.

    **Notes**:

    The model is fully analytic in the latent space, enabling closed-form EM and parameter conversions.
    The shared loading matrix and noise covariance across components distinguish MFA from other mixture models
    and make it particularly suited for modeling data with multiple Gaussian clusters in a lower-dimensional
    latent space.
    """

    # Fields

    obs_dim: int
    """Dimension of observable variables."""

    obs_rep: type[ObsRep]
    """Covariance structure of observable variables."""

    lat_dim: int
    """Dimension of latent factors."""

    n_categories: int
    """Number of mixture components."""

    # Helper properties for latent manifolds

    @property
    @override
    def lat_man(self) -> AnalyticMixture[FullNormal]:
        """Posterior latent manifold: analytic mixture with restricted covariance.

        Uses the restricted covariance structure (PostRep, e.g., diagonal) for computational
        efficiency, since the posterior is evaluated frequently during inference for each data point.
        """
        return AnalyticMixture(
            Normal(self.lat_dim, PositiveDefinite), self.n_categories
        )

    @property
    def mix_hrm(
        self,
    ) -> AnalyticMixture[NormalAnalyticLGM[ObsRep]]:
        """Reinterpret MFA as a Mixture[LGM].

        This provides access to all the mixture machinery (conjugation parameters,
        observables, sampling) by viewing the MFA as a mixture of K independent
        factor analyzers, where the interaction matrix encodes the component-specific
        parameters.
        """
        # Create a single FA manifold with the same dimensions and covariance structures
        # Use replace-like approach for frozen dataclass
        lgm_man: Any = NormalAnalyticLGM(
            obs_dim=self.obs_dim,
            obs_rep=self.obs_rep,
            lat_dim=self.lat_dim,
        )

        return AnalyticMixture(
            lgm_man,
            self.n_categories,
        )

    # Abstract properties from Harmonium

    @property
    @override
    def int_rep(self) -> Rectangular:
        """Matrix representation of interaction matrix."""
        return Rectangular()

    @property
    @override
    def int_obs_emb(self) -> GeneralizedGaussianLocationEmbedding[Normal[ObsRep]]:
        """Embedding of Euclidean interactions into observable space."""
        return GeneralizedGaussianLocationEmbedding(Normal(self.obs_dim, self.obs_rep))

    @property
    @override
    def int_pst_emb(
        self,
    ) -> LinearEmbedding[Euclidean, AnalyticMixture[FullNormal]]:
        """Embedding of Euclidean interactions into posterior latent space."""
        return LinearComposedEmbedding(
            ObservableEmbedding(self.pst_man),
            GeneralizedGaussianLocationEmbedding(
                Normal(self.lat_dim, PositiveDefinite)
            ),
        )

    @override
    def conjugation_parameters(
        self,
        lkl_params: Any,  # Point[Natural, AffineMap[...]]
    ) -> Any:  # Point[Natural, Mixture[FullNormal, Normal[PostRep]]]
        """Compute conjugation parameters for the mixture of factor analyzers.

        Uses the harmonium_mixture_conjugation_parameters decomposition to compute
        the conjugation parameters in the (Y, K) latent space.
        """
        raise NotImplementedError

    # Methods
