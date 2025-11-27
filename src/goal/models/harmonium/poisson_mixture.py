"""Poisson mixture models for count data.

This module provides mixture models specialized for count data, with support for both
standard Poisson distributions (constant variance) and Conway-Maxwell-Poisson (COM-Poisson)
distributions (flexible dispersion).

**Model types**:

- **Poisson Mixture**: Each mixture component is an independent Poisson population where
  each unit has its own rate parameter. Suitable when variance approximately equals mean.

- **COM-Poisson Mixture**: Each component is an independent COM-Poisson population with both
  rate and dispersion parameters. The dispersion parameters are shared across components while
  rates vary. Suitable for overdispersed or underdispersed count data.

**Structure**: Both mixture types are implemented using the harmonium framework with:

- Observable: Product of Poisson or COM-Poisson distributions
- Latent: Categorical distribution over mixture components
- Interaction matrix: Encodes component-specific parameters (rates for Poisson, rates and dispersions for COM-Poisson)

**Type aliases** provide convenient configurations for common use cases:
- `PoissonPopulation`: Analytic product of independent Poisson units
- `PoissonMixture`: Analytic mixture of Poisson populations
- `CoMPoissonMixture`: Mixture of COM-Poisson populations with shared dispersion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticProduct,
    DifferentiableProduct,
    LocationShape,
    Product,
    TupleEmbedding,
)
from ..base.poisson import CoMPoisson, CoMShape, Poisson
from .mixture import AnalyticMixture, Mixture

### Count Mixtures ###

# Type synonyms for common models
type PoissonPopulation = AnalyticProduct[Poisson]
type PopulationShape = Product[CoMShape]
type PoissonMixture = AnalyticMixture[PoissonPopulation]
type CoMPoissonMixture = Mixture[CoMPoissonPopulation, PoissonPopulation]

## Factories ##


@dataclass(frozen=True)
class CoMPoissonPopulation(
    DifferentiableProduct[CoMPoisson], LocationShape[PoissonPopulation, PopulationShape]
):
    """A population of independent Conway-Maxwell-Poisson (COM-Poisson) units.

    The COM-Poisson distribution generalizes the Poisson distribution by introducing a
    dispersion parameter :math:`\\nu`, enabling flexible modeling of count data with
    variance different from the mean.

    For :math:`n` independent COM-Poisson units, the joint density takes the form:

    .. math::

        p(x; \\mu, \\nu) = \\prod_{i=1}^n \\frac{\\mu_i^{x_i}}{(x_i!)^{\\nu_i} Z(\\mu_i, \\nu_i)}

    where for each unit :math:`i`:

    - :math:`\\mu_i > 0` is the rate parameter (mode of the distribution)
    - :math:`\\nu_i > 0` is the dispersion parameter:
        - :math:`\\nu_i = 1`: standard Poisson distribution
        - :math:`\\nu_i > 1`: underdispersed (variance :math:`< \\mu_i`)
        - :math:`\\nu_i < 1`: overdispersed (variance :math:`> \\mu_i`)
    - :math:`Z(\\mu_i, \\nu_i) = \\sum_{j=0}^{\\infty} \\frac{\\mu_i^j}{(j!)^{\\nu_i}}` is the normalizing constant

    **Usage in mixtures**: When used as a mixture component, the rate parameters :math:`\\mu_i` vary
    across mixture components while the dispersion parameters :math:`\\nu_i` are shared. This enables
    flexible mixture models where each component captures both the mode and over/underdispersion
    characteristics of subpopulations.

    **Implementation note**: This class is a `LocationShape` split, where locations are Poisson
    rates and shapes are COM-Poisson dispersion parameters. This enables efficient parameterization
    in mixture models where dispersion is shared.
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
    def join_coords(
        self,
        fst_coords: Array,
        snd_coords: Array,
    ) -> Array:
        """Join location and shape parameters.

        Parameters
        ----------
        first : Array
            Natural parameters for Poisson location (rates).
        second : Array
            Natural parameters for shape (dispersion).

        Returns
        -------
        Array
            Natural parameters for COM-Poisson population.
        """
        params_matrix = jnp.stack([fst_coords, snd_coords])
        return params_matrix.T.reshape(-1)

    @override
    def split_coords(
        self,
        coords: Array,
    ) -> tuple[Array, Array]:
        """Split into location and shape parameters.

        Parameters
        ----------
        params : Array
            Natural parameters for COM-Poisson population.

        Returns
        -------
        tuple[Array, Array]
            Location parameters (Poisson rates) and shape parameters (dispersion).
        """
        matrix = coords.reshape(self.n_reps, 2).T
        loc_params = matrix[0]
        shp_params = matrix[1]
        return loc_params, shp_params


@dataclass(frozen=True)
class PopulationLocationEmbedding(
    TupleEmbedding[
        PoissonPopulation,
        CoMPoissonPopulation,
    ]
):
    """Embedding that projects COM-Poisson to Poisson via location parameters.

    For a replicated location-shape manifold with parameters :math:`((l_1,s_1),\\ldots,(l_n,s_n))`,
    this embedding projects to just the location parameters: :math:`(l_1,\\ldots,l_n)`.

    **Purpose**: In mixture models, components may need different rate parameters (locations) but
    shared dispersion parameters (shapes). This embedding enables such structure by:
    - Projecting COM-Poisson parameters down to their rate components (Poisson)
    - Embedding Poisson rates up to full COM-Poisson by combining with shared dispersions

    **Mathematical structure**: The embedding maps between:
    - Sub-manifold: :math:`\\text{Poisson}^n` (location parameters only)
    - Ambient manifold: :math:`\\text{CoMPoisson}^n` (location-shape pairs)

    The first component of each tuple is extracted or preserved during transformations:
    - Projection: :math:`((l_1, s_1), \\ldots, (l_n, s_n)) \\mapsto (l_1, \\ldots, l_n)`
    - Embedding: :math:`(l_1, \\ldots, l_n) + \\text{shapes} \\mapsto ((l_1, s_1), \\ldots, (l_n, s_n))`

    This enables efficient parameterization in mixture models where component-specific behavior
    comes from varying rates while uniform dispersion is maintained across components.
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


# Constructor-like factory functions for common instances
def poisson_mixture(n_neurons: int, n_components: int) -> PoissonMixture:
    """Create a mixture of independent Poisson populations.

    Parameters
    ----------
    n_neurons : int
        Number of independent Poisson units in each population
    n_components : int
        Number of mixture components

    Returns
    -------
    PoissonMixture
        An analytic mixture of Poisson populations
    """
    pop_man = AnalyticProduct(Poisson(), n_reps=n_neurons)
    return AnalyticMixture(pop_man, n_components)


def com_poisson_mixture(n_neurons: int, n_components: int) -> CoMPoissonMixture:
    """Create a COM-Poisson mixture with shared dispersion parameters.

    In this model, the location parameters vary across mixture components,
    but the dispersion parameters are shared across all components.

    Parameters
    ----------
    n_neurons : int
        Number of independent COM-Poisson units in each population
    n_components : int
        Number of mixture components

    Returns
    -------
    CoMPoissonMixture
        A differentiable mixture with COM-Poisson components
    """
    subspace = PopulationLocationEmbedding(n_neurons)
    return Mixture(n_components, subspace)
