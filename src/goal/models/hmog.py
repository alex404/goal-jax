"""Hierarchical mixture of Gaussians (HMoG) implemented as a hierarchical harmonium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from ..geometry import (
    ComposedSubspace,
    HierarchicalBackward,
    HierarchicalForward,
    LinearMap,
    LocationSubspace,
    PositiveDefinite,
    Rectangular,
    TripleSubspace,
)
from .lgm import LinearGaussianModel
from .mixture import BackwardMixture, ForwardMixture
from .normal import Euclidean, FullNormal, Normal, NormalSubspace

Rep = TypeVar("Rep", bound=PositiveDefinite)


@dataclass(frozen=True)
class ForwardHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    HierarchicalForward[
        Rectangular,
        Normal[ObsRep],
        Euclidean,
        Euclidean,
        ForwardMixture[FullNormal, Normal[LatRep]],
    ],
):
    """A hierarchical mixture of Gaussians implemented as a three-layer harmonium.

    Structure:
    - Observable layer: Gaussian with covariance structure specified by ObsRep
    - First latent layer: Mixture of full-covariance Gaussians
    - Second latent layer: Mixture model over parameters of first latent layer

    Args:
        obs_dim: Dimension of observable space
        lat_dim: Dimension of latent Gaussian components
        n_components: Number of mixture components
        n_subcomponents: Number of submixture components
    """

    # Lower harmonium between X and Y
    _lwr_man: LinearGaussianModel[ObsRep]

    def __init__(
        self,
        obs_dim: int,
        obs_rep: type[ObsRep],
        lat_dim: int,
        lat_rep: type[LatRep],
        n_components: int,
    ):
        obs_man = Normal(obs_dim, obs_rep)
        lwr_lat_man = Normal(lat_dim)
        lwr_sub_lat_man = Normal(lat_dim, lat_rep)
        lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
        object.__setattr__(self, "_lwr_man", lwr_man)
        int_man = LinearMap(
            Rectangular(),
            lwr_lat_man.loc_man,
            obs_man.loc_man,
        )
        nrm_sub = NormalSubspace(lwr_lat_man, lwr_sub_lat_man)
        lat_man = ForwardMixture(lwr_lat_man, n_components, nrm_sub)
        obs_sub = LocationSubspace(obs_man)
        lat_sub = ComposedSubspace(
            TripleSubspace(lat_man), LocationSubspace(lwr_lat_man)
        )
        super().__init__(
            obs_man,
            int_man,
            lat_man,
            obs_sub,
            lat_sub,
        )

    @property
    def lwr_man(
        self,
    ) -> LinearGaussianModel[ObsRep]:
        return self._lwr_man


@dataclass(frozen=True)
class BackwardHMoG[ObsRep: PositiveDefinite](
    HierarchicalBackward[
        Rectangular,
        Normal[ObsRep],
        Euclidean,
        Euclidean,
        BackwardMixture[FullNormal],
    ],
):
    """HMoG where submixture components have full covariance (current impl)."""

    # Lower harmonium between X and Y
    _lwr_man: LinearGaussianModel[ObsRep]

    def __init__(
        self,
        obs_dim: int,
        obs_rep: type[ObsRep],
        lat_dim: int,
        n_components: int,
    ):
        obs_man = Normal(obs_dim, obs_rep)
        lwr_lat_man = Normal(lat_dim)
        lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
        object.__setattr__(self, "_lwr_man", lwr_man)
        int_man = LinearMap(
            Rectangular(),
            lwr_lat_man.loc_man,
            obs_man.loc_man,
        )
        lat_man = BackwardMixture(lwr_lat_man, n_components)
        obs_sub = LocationSubspace(obs_man)
        lat_sub = ComposedSubspace(
            TripleSubspace(lat_man), LocationSubspace(lwr_lat_man)
        )
        super().__init__(
            obs_man,
            int_man,
            lat_man,
            obs_sub,
            lat_sub,
        )

    @property
    def lwr_man(
        self,
    ) -> LinearGaussianModel[ObsRep]:
        return self._lwr_man
