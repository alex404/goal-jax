"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..exponential_family import (
    Backward,
    Mean,
    Natural,
)
from ..exponential_family.distributions import Categorical
from ..manifold import Point
from ..transforms import LinearMap, Rectangular
from .core import BackwardConjugated


@dataclass(frozen=True)
class Mixture[O: Backward](
    BackwardConjugated[Rectangular, O, Categorical],
):
    """A mixture model implemented as a harmonium with categorical latent variables.

    The joint distribution takes the form:

    $$p(x,z) \\propto e^{\\theta_x \\cdot s_x(x)} e^{\\theta_z \\cdot s_z} e^{s_x(x) \\cdot \\Theta_{xz} \\cdot s_z}$$

    where $s_z$ is the one-hot encoding of the categorical variable z.

    This can be rewritten as:

    $$p(x,z=k) \\propto e^{\\theta_x \\cdot s_x(x)} e^{\\theta_k} e^{s_x(x) \\cdot \\theta_{k}}$$

    where $\\theta_k$ and $\\theta_{k}$ are the k-th components of $\\theta_z$ and $\\Theta_{xz}$ respectively.
    """

    def __init__(self, obs_man: O, n_categories: int):
        cat_man = Categorical(n_categories)
        super().__init__(obs_man, cat_man, LinearMap(Rectangular(), cat_man, obs_man))

    def conjugation_parameters(
        self,
        obs_bias: Point[Natural, O],
        int_mat: Point[Natural, LinearMap[Rectangular, Categorical, O]],
    ) -> tuple[Array, Point[Natural, Categorical]]:
        """Compute conjugation parameters for categorical mixture.

        For a categorical mixture model:

        - $\\rho_0 = \\psi(\\theta_x)$
        - $\\rho_z = \\{\\psi(\\theta_x + \\theta_{xz,k}) - \\rho_0\\}_k$
        """
        # Compute base term from observable bias
        rho_0 = self.obs_man.log_partition_function(obs_bias)

        # Get component parameter vectors as columns of interaction matrix
        int_cols = self.inter_man.to_columns(int_mat)

        # Compute adjusted partition function for each component
        rho_z_components = []
        for comp_params in int_cols:
            # For each component k, compute psi(theta_x + theta_xz_k)
            adjusted_obs = obs_bias + comp_params
            comp_term = self.obs_man.log_partition_function(adjusted_obs) - rho_0
            rho_z_components.append(comp_term)

        return rho_0, Point(jnp.array(rho_z_components))

    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> tuple[
        Point[Natural, O], Point[Natural, LinearMap[Rectangular, Categorical, O]]
    ]:
        """Map mean harmonium parameters to natural likelihood parameters.

        For categorical mixtures, we:
        1. Recover weights and component mean parameters
        2. Convert each component mean parameter to natural parameters
        3. Extract base and interaction terms
        """
        comp_means, _ = self.split_mean_mixture(p)
        nat_comps = [self.obs_man.to_natural(comp) for comp in comp_means]
        obs_bias = nat_comps[0]
        int_cols = [comp - obs_bias for comp in nat_comps[1:]]
        int_mat = self.inter_man.from_columns(int_cols)

        return obs_bias, int_mat

    def join_mean_mixture(
        self,
        components: list[Point[Mean, O]],
        weights: Point[Mean, Categorical],
    ) -> Point[Mean, Self]:
        """Create a mixture model in mean coordinates from components and weights."""
        probs = self.lat_man.to_probs(weights)

        weighted_comps: list[Point[Mean, O]] = [
            p * comp for p, comp in zip(probs, components)
        ]

        obs_mean: Point[Mean, O] = reduce(add, weighted_comps)
        int_mat = self.inter_man.from_columns([comp for comp in weighted_comps[1:]])

        return self.join_params(obs_mean, weights, int_mat)

    def split_mean_mixture(
        self, p: Point[Mean, Self]
    ) -> tuple[list[Point[Mean, O]], Point[Mean, Categorical]]:
        """Split a mixture model in mean coordinates into components and weights."""
        obs_mean, cat_mean, int_mat = self.split_params(p)

        probs = self.lat_man.to_probs(cat_mean)
        int_cols = self.inter_man.to_columns(int_mat)

        components: list[Point[Mean, O]] = [
            (obs_mean - reduce(add, int_cols)) / probs[0]
        ]

        for i, col in enumerate(int_cols):
            comp_mean = col / probs[i + 1].item()
            components.append(comp_mean)

        return components, cat_mean

    def join_natural_mixture(
        self,
        components: list[Point[Natural, O]],
        prior: Point[Natural, Categorical],
    ) -> Point[Natural, Self]:
        """Create a mixture model in natural coordinates from components and prior."""

        obs_bias = components[0]
        int_cols = [comp - obs_bias for comp in components[1:]]
        int_mat = self.inter_man.from_columns(int_cols)
        _, rho = self.conjugation_parameters(obs_bias, int_mat)
        lat_bias = prior - rho

        return self.join_params(obs_bias, lat_bias, int_mat)

    def split_natural_mixture(
        self, p: Point[Natural, Self]
    ) -> tuple[list[Point[Natural, O]], Point[Natural, Categorical]]:
        """Split a mixture model in natural coordinates into components and prior."""

        obs_bias, lat_bias, int_mat = self.split_params(p)
        _, rho = self.conjugation_parameters(obs_bias, int_mat)
        int_cols = self.inter_man.to_columns(int_mat)
        components = [obs_bias]  # First component

        for col in int_cols:
            comp = obs_bias + col
            components.append(comp)

        prior = lat_bias + rho

        return components, prior
