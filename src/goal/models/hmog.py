"""Hierarchical mixture of Gaussians (HMoG) implemented as a hierarchical harmonium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from ..geometry import (
    AffineMap,
    HierarchicalHarmonium,
    LinearMap,
    LocationSubspace,
    PairPairSubspace,
    PositiveDefinite,
    Rectangular,
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
        lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
        with lwr_man as lm:
            lkl_man = AffineMap(
                Rectangular(), lm.lat_man.loc_man, LocationSubspace(lm.obs_man)
            )
            upr_man = Mixture(lm.lat_man, n_components)
            lat_sub: PairPairSubspace[
                FullNormal, LinearMap[Rectangular, FullNormal, FullNormal], FullNormal
            ] = PairPairSubspace(upr_man)
            super().__init__(
                lkl_man,
                upr_man,
                lat_sub,
                lwr_man,
            )
