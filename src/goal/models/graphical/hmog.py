"""Hierarchical mixture of Gaussians (HMoG) implemented as a hierarchical harmonium."""

from __future__ import annotations

from ...geometry import (
    DifferentiableHierarchical,
    LinearMap,
    PositiveDefinite,
    Rectangular,
)
from ..base.gaussian.normal import Euclidean, FullNormal, Normal
from .lgm import LinearGaussianModel
from .mixture import DifferentiableMixture

type DifferentiableHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite] = (
    DifferentiableHierarchical[
        Rectangular,
        Normal[ObsRep],
        Euclidean,
        Euclidean,
        DifferentiableMixture[FullNormal, Normal[LatRep]],
    ]
)


def make_differentiable_hmog[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    lat_rep: type[LatRep],
    n_components: int,
) -> DifferentiableHMoG[ObsRep, LatRep]:
    obs_man = Normal(obs_dim, obs_rep)
    lwr_lat_man = Normal(lat_dim)
    lwr_sub_lat_man = Normal(lat_dim, lat_rep)
    lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
    int_man = LinearMap(
        Rectangular(),
        lwr_lat_man.loc_man,
        obs_man.loc_man,
    )
    nrm_sub = NormalSubspace(lwr_lat_man, lwr_sub_lat_man)
    lat_man = DifferentiableMixture(lwr_lat_man, n_components, nrm_sub)
    obs_sub = LocationSubspace(obs_man)
    lat_sub = ComposedSubspace(TripleSubspace(lat_man), LocationSubspace(lwr_lat_man))

    return DifferentiableHierarchical(
        obs_man,
        int_man,
        lat_man,
        obs_sub,
        lat_sub,
    )


# @dataclass(frozen=True)
# class DifferentiableHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
#     DifferentiableHierarchical[
#         Rectangular,
#         Normal[ObsRep],
#         Euclidean,
#         Euclidean,
#         DifferentiableMixture[FullNormal, Normal[LatRep]],
#     ],
# ):
#     """A hierarchical mixture of Gaussians implemented as a three-layer harmonium.
#
#     Structure:
#     - Observable layer: Gaussian with covariance structure specified by ObsRep
#     - First latent layer: Mixture of full-covariance Gaussians
#     - Second latent layer: Mixture model over parameters of first latent layer
#
#     Args:
#         obs_dim: Dimension of observable space
#         lat_dim: Dimension of latent Gaussian components
#         n_components: Number of mixture components
#     """
#
#     # Lower harmonium between X and Y
#     _lwr_man: LinearGaussianModel[ObsRep]
#
#     def __init__(
#         self,
#         obs_dim: int,
#         obs_rep: type[ObsRep],
#         lat_dim: int,
#         lat_rep: type[LatRep],
#         n_components: int,
#     ):
#         obs_man = Normal(obs_dim, obs_rep)
#         lwr_lat_man = Normal(lat_dim)
#         lwr_sub_lat_man = Normal(lat_dim, lat_rep)
#         lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
#         object.__setattr__(self, "_lwr_man", lwr_man)
#         int_man = LinearMap(
#             Rectangular(),
#             lwr_lat_man.loc_man,
#             obs_man.loc_man,
#         )
#         nrm_sub = NormalSubspace(lwr_lat_man, lwr_sub_lat_man)
#         lat_man = DifferentiableMixture(lwr_lat_man, n_components, nrm_sub)
#         obs_sub = LocationSubspace(obs_man)
#         lat_sub = ComposedSubspace(
#             TripleSubspace(lat_man), LocationSubspace(lwr_lat_man)
#         )
#         super().__init__(
#             obs_man,
#             int_man,
#             lat_man,
#             obs_sub,
#             lat_sub,
#         )
#
#     @property
#     @override
#     def lwr_man(
#         self,
#     ) -> LinearGaussianModel[ObsRep]:
#         return self._lwr_man


# @dataclass(frozen=True)
# class AnalyticHMoG[ObsRep: PositiveDefinite](
#     AnalyticHierarchical[
#         Rectangular,
#         Normal[ObsRep],
#         Euclidean,
#         Euclidean,
#         AnalyticMixture[FullNormal],
#     ],
# ):
#     """HMoG where submixture components have full covariance (current impl)."""
#
#     # Lower harmonium between X and Y
#     _lwr_man: LinearGaussianModel[ObsRep]
#
#     def __init__(
#         self,
#         obs_dim: int,
#         obs_rep: type[ObsRep],
#         lat_dim: int,
#         n_components: int,
#     ):
#         obs_man = Normal(obs_dim, obs_rep)
#         lwr_lat_man = Normal(lat_dim)
#         lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
#         object.__setattr__(self, "_lwr_man", lwr_man)
#         int_man = LinearMap(
#             Rectangular(),
#             lwr_lat_man.loc_man,
#             obs_man.loc_man,
#         )
#         lat_man = AnalyticMixture(lwr_lat_man, n_components)
#         obs_sub = LocationSubspace(obs_man)
#         lat_sub = ComposedSubspace(
#             TripleSubspace(lat_man), LocationSubspace(lwr_lat_man)
#         )
#         super().__init__(
#             obs_man,
#             int_man,
#             lat_man,
#             obs_sub,
#             lat_sub,
#         )
#
#     @property
#     @override
#     def lwr_man(
#         self,
#     ) -> LinearGaussianModel[ObsRep]:
#         return self._lwr_man
