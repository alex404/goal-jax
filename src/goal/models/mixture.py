"""Mixture Models as Conjugate Harmoniums.

This module implements mixture models using a harmonium structure where

- The observable manifold is any exponential family with a backward mapping, and
- the Latent Manifold is Categorical distribution over mixture components

Key Features:

- Conjugate structure enabling exact inference
- Support for any exponential family observations
- Parameter conversion between natural and mean coordinates
- Methods for component-wise manipulation
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..geometry import (
    AffineMap,
    Backward,
    BackwardConjugated,
    IdentitySubspace,
    LinearMap,
    Mean,
    Natural,
    Point,
    Rectangular,
)
from .univariate import (
    Categorical,
)


@dataclass(frozen=True)
class Mixture[Observable: Backward](
    BackwardConjugated[Rectangular, Observable, Observable, Categorical, Categorical],
):
    """Mixture model with categorical latent variables.

    Parameters:
        obs_man: Base exponential family for observations
        n_categories: Number of mixture components
    """

    def __init__(self, obs_man: Observable, n_categories: int):
        cat_man = Categorical(n_categories)
        int_man = LinearMap(Rectangular(), cat_man, obs_man)
        obs_sub = IdentitySubspace(obs_man)
        lat_sub = IdentitySubspace(cat_man)
        super().__init__(
            obs_man,
            int_man,
            cat_man,
            obs_sub,
            lat_sub,
        )

    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Categorical, Observable, Observable]
        ],
    ) -> tuple[Array, Point[Natural, Categorical]]:
        """Compute conjugation parameters for categorical mixture.

        For a categorical mixture model:

        - $\\chi = \\psi(\\theta_x)$
        - $\\rho_k = \\psi(\\theta_x + \\theta_{xz,k}) - \\chi$
        """
        # Compute base term from observable bias
        obs_bias, int_mat = self.lkl_man.split_params(lkl_params)
        rho_0 = self.obs_man.log_partition_function(obs_bias)

        # Get component parameter vectors as columns of interaction matrix
        int_cols = self.int_man.to_columns(int_mat)

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
    ) -> Point[Natural, AffineMap[Rectangular, Categorical, Observable, Observable]]:
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
        int_mat = self.int_man.from_columns(int_cols)

        return self.lkl_man.join_params(obs_bias, int_mat)

    def join_mean_mixture(
        self,
        components: list[Point[Mean, Observable]],
        weights: Point[Mean, Categorical],
    ) -> Point[Mean, Self]:
        """Create a mixture model in mean coordinates from components and weights."""
        probs = self.lat_man.to_probs(weights)

        weighted_comps: list[Point[Mean, Observable]] = [
            p * comp for p, comp in zip(probs, components)
        ]

        obs_means: Point[Mean, Observable] = reduce(add, weighted_comps)
        int_means = self.int_man.from_columns([comp for comp in weighted_comps[1:]])

        return self.join_params(obs_means, int_means, weights)

    def split_mean_mixture(
        self, p: Point[Mean, Self]
    ) -> tuple[list[Point[Mean, Observable]], Point[Mean, Categorical]]:
        """Split a mixture model in mean coordinates into components and weights."""
        obs_means, int_means, cat_means = self.split_params(p)

        probs = self.lat_man.to_probs(cat_means)
        int_cols = self.int_man.to_columns(int_means)

        components: list[Point[Mean, Observable]] = [
            (obs_means - reduce(add, int_cols)) / probs[0]
        ]

        for i, col in enumerate(int_cols):
            comp_mean = col / probs[i + 1]
            components.append(comp_mean)

        return components, cat_means

    def join_natural_mixture(
        self,
        components: list[Point[Natural, Observable]],
        prior: Point[Natural, Categorical],
    ) -> Point[Natural, Self]:
        """Create a mixture model in natural coordinates from components and prior."""

        obs_bias = components[0]
        int_cols = [comp - obs_bias for comp in components[1:]]
        int_mat = self.int_man.from_columns(int_cols)
        lkl_params = self.lkl_man.join_params(obs_bias, int_mat)

        return self.join_conjugated(lkl_params, prior)

    def split_natural_mixture(
        self, p: Point[Natural, Self]
    ) -> tuple[list[Point[Natural, Observable]], Point[Natural, Categorical]]:
        """Split a mixture model in natural coordinates into components and prior."""

        lkl_params, prr_params = self.split_conjugated(p)
        obs_bias, int_mat = self.lkl_man.split_params(lkl_params)
        int_cols = self.int_man.to_columns(int_mat)
        components = [obs_bias]  # First component

        for col in int_cols:
            comp = obs_bias + col
            components.append(comp)

        return components, prr_params
