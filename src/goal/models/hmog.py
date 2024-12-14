"""Hierarchical mixture of Gaussians (HMoG) implemented as a hierarchical harmonium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from ..geometry import (
    ComposedSubspace,
    HierarchicalHarmonium,
    LinearMap,
    LocationSubspace,
    PositiveDefinite,
    Rectangular,
    TripleSubspace,
)
from .lgm import LinearGaussianModel
from .mixture import Mixture
from .normal import Euclidean, FullNormal, Normal
from .univariate import Categorical

Rep = TypeVar("Rep", bound=PositiveDefinite)


@dataclass(frozen=True)
class HierarchicalMixtureOfGaussians[ObsRep: PositiveDefinite](
    HierarchicalHarmonium[
        Rectangular,  # Matrix representation
        Normal[ObsRep],  # Observable distribution
        Euclidean,  # Observable subspace
        Euclidean,  # Lower latent subspace
        FullNormal,  # Lower latent distribution
        FullNormal,  # Upper latent subspace
        Categorical,  # Upper latent distribution
        Categorical,  # Upper latent subspace
    ]
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

    def __init__(
        self, obs_dim: int, obs_rep: type[ObsRep], lat_dim: int, n_components: int
    ):
        obs_man = Normal(obs_dim, obs_rep)
        lwr_lat_man = Normal(lat_dim)
        lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
        int_man = LinearMap(
            Rectangular(),
            lwr_lat_man.loc_man,
            obs_man.loc_man,
        )
        lat_man = Mixture(lwr_lat_man, n_components)
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
            lwr_man,
        )
