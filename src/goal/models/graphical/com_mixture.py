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
from typing import Self

import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    AnalyticReplicated,
    DifferentiableReplicated,
    Natural,
    Point,
    Replicated,
    ReplicatedLocationSubspace,
)
from goal.models.base.poisson import CoMPoisson, CoMShape, Poisson
from goal.models.graphical.mixture import DifferentiableMixture

type PoissonPopulation = Replicated[Poisson]
type PopulationShape = Replicated[CoMShape]


@dataclass(frozen=True)
class CoMPoissonPopulation(DifferentiableReplicated[CoMPoisson]):
    """A population of independent COM-Poisson units.

    For $n$ independent COM-Poisson units, the joint density takes the form:

    $$p(x; \\mu, \\nu) = \\prod_{i=1}^n \\frac{\\mu_i^{x_i}}{(x_i!)^{\\nu_i} Z(\\mu_i, \\nu_i)}$$

    where for each unit $i$:
    - $\\mu_i$ is the mode parameter
    - $\\nu_i$ is the dispersion parameter controlling variance relative to Poisson
    - $Z(\\mu, \\nu)$ is the normalizing constant
    """

    def __init__(self, n_repeats: int):
        """Initialize COM-Poisson population.

        Args:
            n_repeats: Number of neurons in the population
        """
        super().__init__(CoMPoisson(), n_repeats)

    @property
    def loc_man(self) -> PoissonPopulation:
        return AnalyticReplicated(Poisson(), n_repeats=self.n_repeats)

    @property
    def shp_man(self) -> PopulationShape:
        return Replicated(CoMShape(), n_repeats=self.n_repeats)

    def join_population_parameters(
        self,
        modes: Point[Natural, PoissonPopulation],
        dispersions: Point[Natural, PopulationShape],
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
        params_matrix = jnp.stack([modes.params, dispersions.params])
        return Point(params_matrix.T.reshape(-1))

    def split_population_parameters(
        self,
        p: Point[Natural, Self],
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
        params_matrix = p.params.reshape(self.n_repeats, 2)
        return Point(params_matrix[:, 0]), Point(params_matrix[:, 1])

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
        theta1 = dispersions * jnp.log(modes)
        theta2 = -dispersions

        params_matrix = jnp.stack([theta1, theta2])
        return Point(params_matrix.T.reshape(-1))

    def split_mode_dispersion(
        self,
        p: Point[Natural, Self],
    ) -> tuple[Array, Array]:
        """Extract mode and dispersion values from population parameters.

        Args:
            p: Natural parameters for the population

        Returns:
            Tuple of (modes, dispersions) arrays with standard parameters
        """
        theta1, theta2 = self.split_population_parameters(p)

        dispersions = -theta2.params
        modes = jnp.exp(-theta1.params / theta2.params)  # μ = exp(-θ₁/θ₂)

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
        params_list = self.split_params(params)
        means = []
        variances = []

        # Compute statistics for each unit
        for unit_params in params_list:
            mean, var = self.man.numerical_mean_variance(unit_params)
            means.append(mean)
            variances.append(var)

        return jnp.array(means), jnp.array(variances)


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
        subspace = ReplicatedLocationSubspace(
            base_location=Poisson(), base_shape=CoMShape(), n_neurons=n_neurons
        )

        super().__init__(obs_man=pop_man, n_categories=n_components, obs_sub=subspace)

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
        comp_means = []
        comp_vars = []
        for comp_params in components:
            mean, var = self.obs_man.numerical_mean_variance(comp_params)
            comp_means.append(mean)
            comp_vars.append(var)

        comp_means = jnp.stack(comp_means)  # Shape: (n_components, n_neurons)
        comp_vars = jnp.stack(comp_vars)  # Shape: (n_components, n_neurons)

        # Compute mixture mean
        mean = jnp.einsum("k,ki->i", weights, comp_means)

        # Compute mixture covariance
        # First term: weighted sum of diagonal component covariances
        cov = jnp.einsum("k,ki->i", weights, comp_vars)
        cov = jnp.diag(cov)

        # Second term: weighted sum of outer products of component means
        mean_outer = jnp.einsum("ki,kj->kij", comp_means, comp_means)
        cov += jnp.einsum("k,kij->ij", weights, mean_outer)

        # Third term: subtract outer product of mixture mean
        cov -= jnp.outer(mean, mean)

        return mean, cov
