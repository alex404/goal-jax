"""Hierarchical mixture of Gaussians (HMoG) implemented as a hierarchical harmonium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ...geometry import (
    AnalyticHierarchical,
    DifferentiableHierarchical,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Subspace,
)
from ..base.gaussian.normal import FullNormal, Normal
from .lgm import LinearGaussianModel
from .mixture import AnalyticMixture, DifferentiableMixture

type DifferentiableHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite] = (
    DifferentiableHierarchical[
        LinearGaussianModel[ObsRep],
        DifferentiableMixture[FullNormal, Normal[LatRep]],
    ]
)

type AnalyticHMoG[ObsRep: PositiveDefinite] = AnalyticHierarchical[
    LinearGaussianModel[ObsRep],
    AnalyticMixture[FullNormal],
]


@dataclass(frozen=True)
class NormalCovarianceSubspace[SubRep: PositiveDefinite, SuperRep: PositiveDefinite](
    Subspace[Normal[SuperRep], Normal[SubRep]]
):
    """Subspace relationship between Normal distributions with different covariance structures.

    This relationship defines how simpler normal distributions (e.g. DiagonalNormal) embed
    into more complex ones (e.g. FullNormal). The key operations are:

    1. Projection: Extract diagonal/scaled components from a full distribution.
       Should be used with mean coordinates (expectations and covariances).

    2. Translation: Embed simpler parameters into the full space.
       Should be used with natural coordinates (natural parameters and precisions).

    For example, a DiagonalNormal can be seen as a submanifold of FullNormal where
    off-diagonal elements are zero.

    Warning:
        This subspace relationship is sensitive to coordinate systems. Projection should only be used with mean coordinates, while translation should only be used with natural coordinates. Incorrect usage will lead to errors.
    """

    # Fields

    _sup_man: Normal[SuperRep]
    _sub_man: Normal[SubRep]

    def __post_init__(self):
        if not isinstance(self.sub_man.cov_man.rep, self.sup_man.cov_man.rep.__class__):
            raise TypeError(
                f"Sub-manifold rep {self.sub_man.cov_man.rep} must be simpler than super-manifold rep {self.sup_man.cov_man.rep}"
            )

    @property
    @override
    def sup_man(self) -> Normal[SuperRep]:
        """Super-manifold."""
        return self._sup_man

    @property
    @override
    def sub_man(self) -> Normal[SubRep]:
        """Sub-manifold."""
        return self._sub_man

    @override
    def project(self, p: Point[Mean, Normal[SuperRep]]) -> Point[Mean, Normal[SubRep]]:
        """Project from super-manifold to sub-manifold.

        This operation is only valid in mean coordinates, where it corresponds to the information projection (moment matching).

        Args:
            p: Point in super-manifold (must be in mean coordinates)

        Returns:
            Projected point in sub-manifold
        """
        return self.sup_man.project_rep(self.sub_man, p)

    @override
    def translate(
        self, p: Point[Natural, Normal[SuperRep]], q: Point[Natural, Normal[SubRep]]
    ) -> Point[Natural, Normal[SuperRep]]:
        """Translate a point in super-manifold by a point in sub-manifold.

        This operation is only valid in natural coordinates, where it embeds the simpler structure into the more complex one before adding, effectively zero padding the missing elements of the point on the submanifold.

        Args:
            p: Point in super-manifold (must be in natural coordinates)
            q: Point in sub-manifold to translate by

        Returns:
            Translated point in super-manifold
        """
        embedded_q = self.sub_man.embed_rep(self.sup_man, q)
        return p + embedded_q


def differentiable_hmog[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    lat_rep: type[LatRep],
    n_components: int,
) -> DifferentiableHMoG[ObsRep, LatRep]:
    mid_lat_man = Normal(lat_dim, PositiveDefinite)
    sub_lat_man = Normal(lat_dim, lat_rep)
    mix_sub = NormalCovarianceSubspace(mid_lat_man, sub_lat_man)
    lwr_hrm = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
    upr_hrm = DifferentiableMixture(mid_lat_man, n_components, mix_sub)

    return DifferentiableHierarchical(
        lwr_hrm,
        upr_hrm,
    )


def analytic_hmog[ObsRep: PositiveDefinite](
    obs_dim: int,
    obs_rep: type[ObsRep],
    lat_dim: int,
    n_components: int,
) -> AnalyticHMoG[ObsRep]:
    mid_lat_man = Normal(lat_dim, PositiveDefinite)
    lwr_hrm = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
    upr_hrm = AnalyticMixture(mid_lat_man, n_components)

    return AnalyticHierarchical(
        lwr_hrm,
        upr_hrm,
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
