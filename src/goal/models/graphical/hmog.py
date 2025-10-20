"""Hierarchical Mixture of Gaussians (HMoG) models."""

from __future__ import annotations

from ...geometry import IdentityEmbedding, PositiveDefinite
from ...geometry.exponential_family.graphical import (
    AnalyticUndirected,
    DifferentiableUndirected,
    SymmetricUndirected,
)
from ..base.gaussian.normal import FullNormal, Normal
from ..harmonium.lgm import (
    NormalAnalyticLinearGaussianModel,
    NormalCovarianceEmbedding,
    NormalDifferentiableLinearGaussianModel,
)
from ..harmonium.mixture import AnalyticMixture, DifferentiableMixture

### HMoGs ###


type DifferentiableHMoG[ObsRep: PositiveDefinite] = (
    DifferentiableUndirected[
        NormalDifferentiableLinearGaussianModel[ObsRep],
        AnalyticMixture[FullNormal],
        DifferentiableMixture[FullNormal, FullNormal],
    ]
)

type SymmetricHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite] = (
    SymmetricUndirected[
        NormalAnalyticLinearGaussianModel[ObsRep],
        DifferentiableMixture[FullNormal, Normal[LatRep]],
    ]
)
type AnalyticHMoG[ObsRep: PositiveDefinite] = AnalyticUndirected[
    NormalAnalyticLinearGaussianModel[ObsRep],
    AnalyticMixture[FullNormal],
]

## Factories ##


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

    return DifferentiableUndirected(
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

    return SymmetricUndirected(
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

    return AnalyticUndirected(
        lwr_hrm,
        upr_hrm,
    )
