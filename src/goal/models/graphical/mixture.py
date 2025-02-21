"""Mixture Models as Conjugate Harmoniums.

This module implements mixture models using a harmonium structure where

- The observable manifold is any exponential family with a backward mapping, and
- the Latent Manifold is Categorical distribution over mixture components

Key Features:

- Conjugate structure enabling exact inference
- Support for any exponential family observations
- Parameter conversion between natural and mean coordinates
- Methods for component-wise manipulation

#### Class Hierarchy

![Class Hierarchy](mixture.svg)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self, override

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AffineMap,
    Analytic,
    AnalyticConjugated,
    Differentiable,
    DifferentiableConjugated,
    ExponentialFamily,
    Harmonium,
    IdentitySubspace,
    Mean,
    Natural,
    Point,
    Product,
    Rectangular,
    StatisticalMoments,
    Subspace,
)
from ..base.categorical import (
    Categorical,
)


@dataclass(frozen=True)
class Mixture[Observable: ExponentialFamily, SubObservable: ExponentialFamily](
    Harmonium[Rectangular, Observable, SubObservable, Categorical, Categorical],
):
    """Mixture models with exponential family observations.

    Parameters:
        n_categories: Number of mixture components
        _obs_sub: Subspace relationship for observable parameters (default: Identity)
    """

    # Fields

    n_categories: int
    _obs_sub: Subspace[Observable, SubObservable]

    # Overrides

    @property
    @override
    def int_rep(self) -> Rectangular:
        return Rectangular()

    @property
    @override
    def obs_sub(self) -> Subspace[Observable, SubObservable]:
        return self._obs_sub

    @property
    @override
    def lat_sub(self) -> IdentitySubspace[Categorical]:
        return IdentitySubspace(Categorical(self.n_categories))

    # Template Methods

    @property
    def cmp_man(self) -> Product[Observable]:
        """Manifold for all components of mixture."""
        return Product(self.obs_man, self.n_categories)

    def join_mean_mixture(
        self,
        components: Point[Mean, Product[Observable]],
        weights: Point[Mean, Categorical],
    ) -> Point[Mean, Self]:
        """Create a mixture model in mean coordinates from components and weights.

        Note: if only a submanifold of interactions on the observables are considered
        (i.e. obs_sub is not the IdentitySubspace), then information in the components
        will be discarded.
        """
        probs = self.lat_man.to_probs(weights)
        probs_shape = [-1] + [1] * (len(self.cmp_man.coordinates_shape) - 1)
        probs = probs.reshape(probs_shape)

        # Scale components - shape: (n_categories, obs_dim)
        weighted_comps = components.array * probs

        # Sum for observable means - shape: (obs_dim,)
        obs_means = self.obs_man.mean_point(jnp.sum(weighted_comps, axis=0))

        cmp_man_minus = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # projected_comps shape: (n_categories-1, sub_obs_dim)
        projected_comps = cmp_man_minus.man_map(
            self.obs_sub.project,
            cmp_man_minus.mean_point(weighted_comps[1:]),
        )
        # int_means shape: (sub_obs_dim, n_categories-1)
        int_means = self.int_man.from_columns(projected_comps)

        return self.join_params(obs_means, int_means, weights)

    def split_mean_mixture(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Mean, Product[Observable]], Point[Mean, Categorical]]:
        """Split a mixture model in mean coordinates into components and weights."""
        obs_means, int_means, cat_means = self.split_params(p)
        probs = self.lat_man.to_probs(cat_means)  # shape: (n_categories,)

        # Get interaction columns - shape: (n_categories-1, obs_dim)
        int_cols = self.int_man.to_columns(int_means)

        # Compute first component
        # Sum interaction columns - shape: (obs_dim,)
        sum_interactions = self.obs_man.mean_point(jnp.sum(int_cols.array, axis=0))
        first_comp = (obs_means - sum_interactions) / probs[0]

        # Scale remaining components by their probabilities
        # shape: (n_categories-1, obs_dim)

        probs_shape = [-1] + [1] * (len(self.cmp_man.coordinates_shape) - 1)
        other_comps = int_cols.array / probs[1:].reshape(probs_shape)

        # Combine components - shape: (n_categories, obs_dim)
        components = jnp.vstack([first_comp.array[None, :], other_comps])

        return self.cmp_man.mean_point(components), cat_means


@dataclass(frozen=True)
class DifferentiableMixture[
    Observable: Differentiable,
    SubObservable: ExponentialFamily,
](
    Mixture[Observable, SubObservable],
    DifferentiableConjugated[
        Rectangular, Observable, SubObservable, Categorical, Categorical
    ],
):
    # Overrides

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Categorical, SubObservable, Observable]
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

        # int_comps shape: (n_categories - 1, sub_obs_dim)
        int_comps = self.int_man.to_columns(int_mat)

        def compute_rho(comp_params: Point[Natural, SubObservable]) -> Array:
            adjusted_obs = self.obs_sub.translate(obs_bias, comp_params)
            return self.obs_man.log_partition_function(adjusted_obs) - rho_0

        # rho_z shape: (n_categories - 1,)
        rho_z = self.int_man.col_man.map(compute_rho, int_comps)

        return rho_0, self.lat_man.natural_point(rho_z)

    # Methods

    def split_natural_mixture(
        self, p: Point[Natural, Self]
    ) -> tuple[Point[Natural, Product[Observable]], Point[Natural, Categorical]]:
        """Split a mixture model in natural coordinates into components and prior."""

        lkl_params, prr_params = self.split_conjugated(p)
        obs_bias, int_mat = self.lkl_man.split_params(lkl_params)

        # into_cols shape: (n_categories - 1, sub_obs_dim)
        int_cols = self.int_man.to_columns(int_mat)

        def translate_col(
            col: Point[Natural, SubObservable],
        ) -> Point[Natural, Observable]:
            return self.obs_sub.translate(obs_bias, col)

        # translated shape: (n_categories - 1, obs_dim)
        translated = self.int_man.col_man.man_map(translate_col, int_cols)
        # components shape: (n_categories, obs_dim)
        components = jnp.vstack([obs_bias.array[None, :], translated.array])

        return self.cmp_man.natural_point(components), prr_params

    def join_natural_mixture(
        self,
        components: Point[Natural, Product[Observable]],
        prior: Point[Natural, Categorical],
    ) -> Point[Natural, Self]:
        """Create a mixture model in natural coordinates from components and prior.

        Parameters in the mixing subspace use the first component as reference/anchor and store differences in the interaction matrix. Parameters outside the mixing subspace use the weighted average of all components.
        """
        # Get probabilities for weighting - shape: (n_categories,)
        probs = self.lat_man.to_probs(self.lat_man.to_mean(prior))
        probs_shape = [-1] + [1] * (len(self.cmp_man.coordinates_shape) - 1)

        # Get anchor (first component) - shape: (obs_dim,)
        anchor = self.cmp_man.get_replicate(components, jnp.asarray(0))
        anchor_sub = self.obs_sub.project(anchor)  # shape: (sub_obs_dim,)

        # Compute weighted average - shape: (obs_dim,)
        avg_params = self.obs_man.natural_point(
            jnp.sum(components.array * probs.reshape(probs_shape), axis=0)
        )

        # Create obs_bas combining in/out of subspace parameters
        anchor_out_sub = self.obs_sub.translate(
            avg_params, -self.obs_sub.project(avg_params)
        )
        anchor_in_sub = self.obs_sub.translate(
            self.obs_man.natural_point(jnp.zeros_like(anchor.array)), anchor_sub
        )
        obs_bias = anchor_in_sub + anchor_out_sub  # shape: (obs_dim,)

        def to_interaction(
            comp: Point[Natural, Observable],
        ) -> Point[Natural, SubObservable]:
            return self.obs_sub.project(comp) - anchor_sub

        cmp_man_minus = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # projected_comps shape: (n_categories-1, sub_obs_dim)
        projected_comps = cmp_man_minus.man_map(
            to_interaction, cmp_man_minus.natural_point(components.array[1:])
        )

        # int_mat shape: (sub_obs_dim, n_categories-1)
        int_mat = self.int_man.from_columns(projected_comps)
        lkl_params = self.lkl_man.join_params(obs_bias, int_mat)

        return self.join_conjugated(lkl_params, prior)

    def observable_mean_covariance(
        self, params: Point[Natural, Self]
    ) -> tuple[Array, Array]:
        """Compute mean and covariance of the observable variables.

        Requires the observable manifold to implement StatisticalMoments.

        Args:
            params: Parameters in natural coordinates

        Returns:
            Tuple of (mean array, covariance array)

        Raises:
            TypeError: If observable manifold doesn't support moment computation
        """
        if not isinstance(self.obs_man, StatisticalMoments):
            raise TypeError(
                f"Observable manifold {type(self.obs_man)} does not support statistical moment computation"
            )

        comp_params, cat_params = self.split_natural_mixture(params)
        weights = self.lat_man.to_probs(self.lat_man.to_mean(cat_params))

        # Use statistical_mean/covariance instead of points to avoid type issues
        cmp_means = self.cmp_man.map(self.obs_man.statistical_mean, comp_params)  # pyright: ignore[reportArgumentType]
        cmp_covs = self.cmp_man.map(self.obs_man.statistical_covariance, comp_params)  # pyright: ignore[reportArgumentType]

        # Compute mixture mean
        mean = jnp.einsum("k,ki->i", weights, cmp_means)

        # Compute mixture covariance using law of total variance
        # First term: weighted sum of component covariances
        cov = jnp.einsum("k,kij->ij", weights, cmp_covs)

        # Second term: weighted sum of outer products of component means
        mean_outer = jnp.einsum("ki,kj->kij", cmp_means, cmp_means)
        cov += jnp.einsum("k,kij->ij", weights, mean_outer)

        # Third term: subtract outer product of mixture mean
        cov -= jnp.outer(mean, mean)

        return mean, cov


@dataclass(frozen=True)
class AnalyticMixture[Observable: Analytic](
    DifferentiableMixture[Observable, Observable],
    AnalyticConjugated[Rectangular, Observable, Observable, Categorical, Categorical],
):
    """Mixture model with analytical entropy.

    This class adds the ability to convert from mean to natural parameters and
    compute negative entropy. Only works with identity subspace relationships.

    Args:
        obs_man: Base exponential family for observations
        n_categories: Number of mixture components
    """

    # Constructor

    def __init__(self, obs_man: Observable, n_categories: int):
        super().__init__(n_categories, IdentitySubspace(obs_man))

    # Overrides

    @override
    def to_natural_likelihood(
        self, params: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rectangular, Categorical, Observable, Observable]]:
        """Map mean harmonium parameters to natural likelihood parameters.

        For categorical mixtures, we:
        1. Recover weights and component mean parameters
        2. Convert each component mean parameter to natural parameters
        3. Extract base and interaction terms
        """
        # Get component means - shape: (n_categories, obs_dim)
        comp_means, _ = self.split_mean_mixture(params)

        # Convert each component to natural parameters
        def to_natural(mean: Point[Mean, Observable]) -> Array:
            return self.obs_man.to_natural(mean).array

        # nat_comps shape: (n_categories, obs_dim)
        nat_comps = self.cmp_man.natural_point(self.cmp_man.map(to_natural, comp_means))

        # Get anchor (first component) - shape: (obs_dim,)
        obs_bias = self.cmp_man.get_replicate(nat_comps, jnp.asarray(0))

        def to_interaction(
            nat: Point[Natural, Observable],
        ) -> Point[Natural, Observable]:
            return nat - obs_bias

        # Convert remaining components to interactions
        cmp_man1 = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # int_cols shape: (n_categories-1, obs_dim)
        int_cols = cmp_man1.man_map(
            to_interaction, cmp_man1.natural_point(nat_comps.array[1:])
        )

        # int_mat shape: (obs_dim, n_categories-1)
        int_mat = self.int_man.from_columns(int_cols)

        return self.lkl_man.join_params(obs_bias, int_mat)  # Methods
