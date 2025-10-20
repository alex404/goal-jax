"""Count mixture models using COM-Poisson populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import jax.numpy as jnp

from ...geometry import (
    AnalyticProduct,
    DifferentiableProduct,
    LocationShape,
    Natural,
    Point,
    Product,
    TupleEmbedding,
)
from ..base.poisson import CoMPoisson, CoMShape, Poisson
from .mixture import AnalyticMixture, DifferentiableMixture

### Count Mixtures ###

# Type synonyms for common models
type PoissonPopulation = AnalyticProduct[Poisson]
type PopulationShape = Product[CoMShape]
type PoissonMixture = AnalyticMixture[PoissonPopulation]
type CoMPoissonMixture = DifferentiableMixture[CoMPoissonPopulation, PoissonPopulation]

## Factories ##


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
class PopulationLocationEmbedding(
    TupleEmbedding[
        PoissonPopulation,
        CoMPoissonPopulation,
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
    def tup_idx(
        self,
    ) -> int:
        return 0

    @property
    @override
    def amb_man(self) -> CoMPoissonPopulation:
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
    subspace = PopulationLocationEmbedding(n_neurons)
    return DifferentiableMixture(n_components, subspace)
