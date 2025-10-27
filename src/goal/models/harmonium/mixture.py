"""Mixture Models as Conjugate Harmoniums.

This module implements mixture models using a harmonium structure where

- The observable manifold is an exponential family, and
- the latent manifold is `Categorical` distribution over mixture components
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
    ExponentialFamily,
    IdentityEmbedding,
    LinearEmbedding,
    Mean,
    Natural,
    Point,
    Product,
    Rectangular,
    StatisticalMoments,
    SymmetricConjugated,
    SymmetricHarmonium,
)
from ..base.categorical import (
    Categorical,
)


@dataclass(frozen=True)
class Mixture[Observable: ExponentialFamily, SubObservable: ExponentialFamily](
    SymmetricHarmonium[
        Rectangular, Observable, SubObservable, Categorical, Categorical
    ],
):
    """Mixture models with exponential family observations.

    A mixture model represents a weighted combination of component distributions:

    $$p(x) = \\sum_{k=1}^K \\pi_k p(x|\\theta_k)$$

    where $\\pi_k$ are mixture weights summing to 1, and $p(x|\\theta_k)$ are component
    distributions from the same exponential family.

    In the harmonium framework, mixture models are implemented using:
        - A categorical latent variable $z$ representing component assignment
        - An observable distribution family for component distributions
        - An interaction matrix between $z$ and $x$ that that contain component specific parameters
    """

    # Fields

    n_categories: int
    """Number of mixture components."""

    _obs_emb: LinearEmbedding[SubObservable, Observable]
    """Embedding for observable manifold."""

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

        **NB:** if only a submanifold of interactions on the observables are considered (i.e. obs_emb is not the IdentityEmbeddingSubspace), then the statistics outside of the interaction submanifold will simply be averaged.
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
            self.obs_emb.project,
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

    # Overrides

    @property
    @override
    def int_rep(self) -> Rectangular:
        return Rectangular()

    @property
    @override
    def obs_emb(self) -> LinearEmbedding[SubObservable, Observable]:
        return self._obs_emb

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self.n_categories)

    @property
    @override
    def int_lat_emb(self) -> IdentityEmbedding[Categorical]:
        return IdentityEmbedding(Categorical(self.n_categories))


@dataclass(frozen=True)
class DifferentiableMixture[
    Observable: Differentiable,
    SubObservable: ExponentialFamily,
](
    Mixture[Observable, SubObservable],
    SymmetricConjugated[
        Rectangular, Observable, SubObservable, Categorical, Categorical
    ],
):
    """Mixture model with differentiable components.

    This implementation adds numerical capabilities:

    - Exact log-partition function calculation
    - Density evaluation for observations
    - Mean and covariance calculation
    - Parameter estimation via posterior statistics/expectation step
    """

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
            return self.obs_emb.translate(obs_bias, col)

        # translated shape: (n_categories - 1, obs_dim)
        translated = self.int_man.col_man.man_map(translate_col, int_cols)
        # components shape: (n_categories, obs_dim)
        components = jnp.vstack([obs_bias.array[None, :], translated.array])

        return self.cmp_man.natural_point(components), prr_params

    def observable_mean_covariance(
        self, params: Point[Natural, Self]
    ) -> tuple[Array, Array]:
        """Compute mean and covariance of the observable variables."""
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

    # Overrides

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Categorical, SubObservable, Observable]
        ],
    ) -> Point[Natural, Categorical]:
        """Compute conjugation parameters for categorical mixture. In particular,

        $$\\rho_k = \\psi(\\theta_X + \\theta_{XZ,k}) - \\psi(\\theta_X)$$
        """
        # Compute base term from observable bias
        obs_bias, int_mat = self.lkl_man.split_params(lkl_params)
        rho_0 = self.obs_man.log_partition_function(obs_bias)

        # int_comps shape: (n_categories - 1, sub_obs_dim)
        int_comps = self.int_man.to_columns(int_mat)

        def compute_rho(comp_params: Point[Natural, SubObservable]) -> Array:
            adjusted_obs = self.obs_emb.translate(obs_bias, comp_params)
            return self.obs_man.log_partition_function(adjusted_obs) - rho_0

        # rho_z shape: (n_categories - 1,)
        rho_z = self.int_man.col_man.map(compute_rho, int_comps)

        return self.lat_man.natural_point(rho_z)


@dataclass(frozen=True)
class AnalyticMixture[Observable: Analytic](
    DifferentiableMixture[Observable, Observable],
    AnalyticConjugated[Rectangular, Observable, Observable, Categorical, Categorical],
):
    """Mixture model with analytically tractable components. Amongst other things, this enables closed-form implementation of expectation-maximization."""

    # Constructor

    def __init__(self, obs_man: Observable, n_categories: int):
        super().__init__(n_categories, IdentityEmbedding(obs_man))

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

    def join_natural_mixture(
        self,
        components: Point[Natural, Product[Observable]],
        prior: Point[Natural, Categorical],
    ) -> Point[Natural, Self]:
        """Create a mixture model in natural coordinates from components and prior.

        Parameters in the mixing subspace use the first component as reference/anchor and store differences in the interaction matrix. Parameters outside the mixing subspace use the weighted average of all components.
        """
        # Get anchor (first component) - shape: (obs_dim,)
        obs_bias = self.cmp_man.get_replicate(components, jnp.asarray(0))

        def to_interaction(
            comp: Point[Natural, Observable],
        ) -> Point[Natural, Observable]:
            return comp - obs_bias

        cmp_man_minus = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # projected_comps shape: (n_categories-1, sub_obs_dim)
        projected_comps = cmp_man_minus.man_map(
            to_interaction, cmp_man_minus.natural_point(components.array[1:])
        )

        # int_mat shape: (sub_obs_dim, n_categories-1)
        int_mat = self.int_man.from_columns(projected_comps)
        lkl_params = self.lkl_man.join_params(obs_bias, int_mat)

        return self.join_conjugated(lkl_params, prior)
