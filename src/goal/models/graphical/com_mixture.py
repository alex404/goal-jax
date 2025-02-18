"""Mixture models for populations of Conway-Maxwell Poisson (COM-Poisson) neurons.

This module implements mixture models for populations of COM-Poisson distributions where
mixture components only affect location (rate) parameters while sharing shape (dispersion)
parameters. The module provides:

1. CoMPoissonPopulation: A structured representation of n independent COM-Poisson units
2. Mixture models that can capture heterogeneous firing rate patterns while maintaining
   shared dispersion structure

The COM-Poisson population density takes the form:

$$p(x; \\mu, \\nu) = \\prod_{i=1}^n \\frac{\\mu_i^{x_i}}{(x_i!)^{\\nu_i} Z(\\mu_i, \\nu_i)}$$

where:
- $\\mu_i$ are the mode parameters for each unit
- $\\nu_i$ are the dispersion parameters
- $Z(\\mu, \\nu)$ is the normalizing constant
"""

from dataclasses import dataclass
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    AnalyticProduct,
    DifferentiableProduct,
    LocationShape,
    LocationSubspace,
    Natural,
    Point,
    Product,
)
from goal.models.base.poisson import CoMPoisson, CoMShape, Poisson
from goal.models.graphical.mixture import DifferentiableMixture

type PoissonPopulation = AnalyticProduct[Poisson]
type PopulationShape = Product[CoMShape]


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
        """Construct population from mode and dispersion parameters.

        Given population parameters:
        - Modes $\\mu = (\\mu_1,\\ldots,\\mu_n)$
        - Dispersions $\\nu = (\\nu_1,\\ldots,\\nu_n)$

        Returns the natural parameters of the joint distribution.

        Args:
            modes: Mode parameters for each unit
            dispersions: Dispersion parameters for each unit

        Returns:
            Natural parameters for the full population
        """
        params_matrix = jnp.stack([first.array, second.array])
        return self.natural_point(params_matrix.T.reshape(-1))

    @override
    def split_params(
        self,
        params: Point[Natural, Self],
    ) -> tuple[Point[Natural, PoissonPopulation], Point[Natural, PopulationShape]]:
        """Split population parameters into modes and dispersions.

        Given natural parameters of the joint distribution, returns:
        - Modes $\\mu = (\\mu_1,\\ldots,\\mu_n)$
        - Dispersions $\\nu = (\\nu_1,\\ldots,\\nu_n)$

        Args:
            p: Natural parameters for the population

        Returns:
            Tuple of (modes, dispersions) for the population
        """
        matrix = params.array.reshape(self.n_reps, 2).T
        loc_params = self.fst_man.natural_point(matrix[0])
        shp_params = self.snd_man.natural_point(matrix[1])
        return loc_params, shp_params

    def join_mode_dispersion(
        self,
        modes: Array,
        dispersions: Array,
    ) -> Point[Natural, Self]:
        """Construct population from mode and dispersion values.

        Args:
            modes: Array of mode parameters $\\mu$ for each unit
            dispersions: Array of dispersion parameters $\\nu$ for each unit

        Returns:
            Natural parameters for the population
        """
        # Convert modes and dispersions to natural parameters
        loc_params = self.fst_man.natural_point(dispersions * jnp.log(modes))
        shp_params = self.snd_man.natural_point(-dispersions)

        return self.join_params(loc_params, shp_params)

    def split_mode_dispersion(
        self,
        params: Point[Natural, Self],
    ) -> tuple[Array, Array]:
        """Extract mode and dispersion values from population parameters.

        Args:
            p: Natural parameters for the population

        Returns:
            Tuple of (modes, dispersions) arrays with standard parameters
        """
        loc_params, shp_params = self.split_params(params)
        loc_array = loc_params.array
        shp_array = shp_params.array

        dispersions = -shp_array
        modes = jnp.exp(-loc_array / shp_array)

        return modes, dispersions

    def numerical_mean_variance(
        self: Self,
        params: Point[Natural, Self],
    ) -> tuple[Array, Array]:
        """Compute mean vector and diagonal variance for population.

        Computes statistics for each neuron independently using numerical integration.

        Args:
            params: Natural parameters for population

        Returns:
            Tuple of (mean vector, variance vector)
        """

        # Compute statistics for each unit
        def rep_fun(array: Array) -> tuple[Array, Array]:
            params = self.rep_man.natural_point(array)
            return self.rep_man.numerical_mean_variance(params)

        return jax.vmap(rep_fun)(params.array)


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

    n_reps: int

    @property
    @override
    def sup_man(self) -> CoMPoissonPopulation:
        return CoMPoissonPopulation(self.n_reps)

    @property
    @override
    def sub_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), n_reps=self.n_reps)


@dataclass(frozen=True)
class CoMMixture(DifferentiableMixture[CoMPoissonPopulation, PoissonPopulation]):
    """Mixture model for COM-Poisson populations where components affect only rate parameters.

    For a population of n neurons, the model defines a mixture distribution:

    $$p(x) = \\sum_{k=1}^K \\pi_k \\prod_{i=1}^n \\frac{\\mu_{k,i}^{x_i}}{(x_i!)^{\\nu_i} Z(\\mu_{k,i}, \\nu_i)}$$

    where:
    - $\\pi_k$ are mixture weights
    - $\\mu_{k,i}$ are component-specific rate parameters
    - $\\nu_i$ are shared dispersion parameters
    - $Z(\\mu, \\nu)$ is the normalizing constant

    This structure allows the model to capture heterogeneous rate patterns while
    maintaining shared dispersion structure across mixture components.
    """

    def __init__(self, n_neurons: int, n_components: int):
        """Initialize COM-Poisson mixture model.

        Args:
            n_neurons: Number of neurons in the population
            n_components: Number of mixture components
        """
        # Create population manifold
        pop_man = CoMPoissonPopulation(n_neurons)

        # Create subspace relationship for rate-only components
        subspace = PopulationLocationSubspace(n_reps=n_neurons)

        super().__init__(pop_man, n_components, subspace)

    def numerical_mean_covariance(
        self,
        params: Point[Natural, Self],
    ) -> tuple[Array, Array]:
        """Compute mean vector and covariance matrix for mixture model numerically.

        For a mixture model with K components:
        $$E[X] = \\sum_{k=1}^K \\pi_k E[X|k]$$
        $$\\text{Cov}(X) = \\sum_{k=1}^K \\pi_k (\\text{Cov}(X|k) + E[X|k]E[X|k]^T) - E[X]E[X]^T$$

        Uses numerical integration to compute component statistics.

        Args:
            params: Natural parameters for mixture model

        Returns:
            Tuple of (mean vector, covariance matrix)
        """
        # Split into observable and latent parameters
        components, cat_params = self.split_natural_mixture(params)
        mix_means = self.lat_man.to_mean(cat_params)

        # Get mixture weights
        weights = self.lat_man.to_probs(mix_means)

        # Compute component means and variances

        def cmp_fun(array: Array) -> tuple[Array, Array]:
            params = self.obs_man.natural_point(array)
            return self.obs_man.numerical_mean_variance(params)

        cmp_means, cmp_vars = jax.vmap(cmp_fun)(components.array)

        # Compute mixture mean
        mean = jnp.einsum("k,ki->i", weights, cmp_means)

        # Compute mixture covariance
        # First term: weighted sum of diagonal component covariances
        cov = jnp.einsum("k,ki->i", weights, cmp_vars)
        cov = jnp.diag(cov)

        # Second term: weighted sum of outer products of component means
        mean_outer = jnp.einsum("ki,kj->kij", cmp_means, cmp_means)
        cov += jnp.einsum("k,kij->ij", weights, mean_outer)

        # Third term: subtract outer product of mixture mean
        cov -= jnp.outer(mean, mean)

        return mean, cov
