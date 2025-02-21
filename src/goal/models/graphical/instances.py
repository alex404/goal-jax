"""Hierarchical mixture of Gaussians (HMoG) implemented as a hierarchical harmonium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import jax.numpy as jnp

from ...geometry import (
    AnalyticHierarchical,
    AnalyticProduct,
    DifferentiableHierarchical,
    DifferentiableProduct,
    LocationShape,
    LocationSubspace,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Product,
    Subspace,
)
from ..base.gaussian.normal import FullNormal, Normal
from ..base.poisson import CoMPoisson, CoMShape, Poisson
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
    upr_hrm = DifferentiableMixture(n_components, mix_sub)

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


# Type synonyms for common models
type PoissonPopulation = AnalyticProduct[Poisson]
type PopulationShape = Product[CoMShape]
type PoissonMixture = AnalyticMixture[PoissonPopulation]
type CoMPoissonMixture = DifferentiableMixture[CoMPoissonPopulation, PoissonPopulation]


@dataclass(frozen=True)
class CoMPoissonPopulation(
    DifferentiableProduct[CoMPoisson], LocationShape[PoissonPopulation, PopulationShape]
):
    """A population of independent COM-Poisson units.

    For $n$ independent COM-Poisson units, the joint density takes the form:

    $$p(x; \\mu, \\nu) = \\prod_{i=1}^n \\frac{\\mu_i^{x_i}}{(x_i!)^{\\nu_i} Z(\\mu_i, \\nu_i)}$$

    where for each unit $i$:
    - $\\mu_i$ is the mode parameter
    - $\\nu_i$ is the dispersion parameter controlling variance relative to Poisson
    - $Z(\\mu, \\nu)$ is the normalizing constant
    """

    def __init__(self, n_reps: int):
        """Initialize COM-Poisson population.

        Args:
            n_reps: Number of neurons in the population
        """
        super().__init__(CoMPoisson(), n_reps)

    @property
    @override
    def fst_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), n_reps=self.n_reps)

    @property
    @override
    def snd_man(self) -> PopulationShape:
        return Product(CoMShape(), n_reps=self.n_reps)

    @override
    def join_params(
        self,
        first: Point[Natural, PoissonPopulation],
        second: Point[Natural, PopulationShape],
    ) -> Point[Natural, Self]:
        params_matrix = jnp.stack([first.array, second.array])
        return self.natural_point(params_matrix.T.reshape(-1))

    @override
    def split_params(
        self,
        params: Point[Natural, Self],
    ) -> tuple[Point[Natural, PoissonPopulation], Point[Natural, PopulationShape]]:
        matrix = params.array.reshape(self.n_reps, 2).T
        loc_params = self.fst_man.natural_point(matrix[0])
        shp_params = self.snd_man.natural_point(matrix[1])
        return loc_params, shp_params


@dataclass(frozen=True)
class PopulationLocationSubspace(
    LocationSubspace[
        CoMPoissonPopulation,
        PoissonPopulation,
    ]
):
    """Subspace relationship that projects only to location parameters of replicated location-shape manifolds.

    For a replicated location-shape manifold with parameters:
    $((l_1,s_1),\\ldots,(l_n,s_n))$

    Projects to just the location parameters:
    $(l_1,\\ldots,l_n)$

    This enables mixture models where components only affect location parameters while
    sharing shape parameters across components.
    """

    n_neurons: int

    @property
    @override
    def sup_man(self) -> CoMPoissonPopulation:
        return CoMPoissonPopulation(n_reps=self.n_neurons)

    @property
    @override
    def sub_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), n_reps=self.n_neurons)


# Constructors for common instances
def poisson_mixture(n_neurons: int, n_components: int) -> PoissonMixture:
    """Create a mixture of independent Poisson populations."""
    pop_man = AnalyticProduct(Poisson(), n_reps=n_neurons)
    return AnalyticMixture(pop_man, n_components)


def com_poisson_mixture(n_neurons: int, n_components: int) -> CoMPoissonMixture:
    """Create a COM-Poisson mixture with shared dispersion parameters."""
    subspace = PopulationLocationSubspace(n_neurons)
    return DifferentiableMixture(n_components, subspace)
