"""Mixture Models as Conjugate Harmoniums.

This module implements mixture models using a harmonium structure where

- The observable manifold is an exponential family, and
- the latent manifold is `Categorical` distribution over mixture components
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    Analytic,
    AnalyticConjugated,
    Differentiable,
    DifferentiableConjugated,
    ExponentialFamily,
    IdentityEmbedding,
    LinearEmbedding,
    Product,
    Rectangular,
    RectangularMap,
    StatisticalMoments,
    SymmetricConjugated,
)
from ..base.categorical import (
    Categorical,
)


@dataclass(frozen=True)
class Mixture[Observable: Differentiable, IntObservable: ExponentialFamily](
    SymmetricConjugated[Observable, IntObservable, Categorical, Categorical],
    DifferentiableConjugated[
        Observable, IntObservable, Categorical, Categorical, Categorical
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

    In this most general formulation, the interaction parameters may be restricted to a submanifold of the observable manifold.
    """

    # Fields

    n_categories: int
    """Number of mixture components."""

    _int_obs_emb: LinearEmbedding[IntObservable, Observable]
    """Embedding for observable manifold."""

    # Template Methods

    @property
    def cmp_man(self) -> Product[Observable]:
        """Manifold for all components of mixture."""
        return Product(self.obs_man, self.n_categories)

    def join_mean_mixture(
        self,
        components: Array,
        weights: Array,
    ) -> Array:
        """Create a mixture model in mean coordinates from components and weights.

        In mean coordinates, projections are the correct operation because expectations
        of sufficient statistics are preserved: $\\mathbb{E}[s_{\\text{sub}}(x)] = \\text{project}(\\mathbb{E}[s_{\\text{full}}(x)])$.
        This method projects component parameters to the interaction submanifold, storing only
        the submanifold portion while averaging out statistics outside it.

        **Technical note:** When $\\text{Observable} \\neq \\text{IntObservable}$, component
        statistics outside the interaction submanifold are irreversibly lost (aggregated into
        observable biases). This is mathematically correct for mean coordinates but makes
        the operation non-invertible without additional information.
        """
        probs = self.pst_man.to_probs(weights)
        # Scale components - shape: (n_categories, obs_dim)
        weighted_comps = components * probs[:, None]

        # Sum for observable means - shape: (obs_dim,)
        obs_means = jnp.sum(weighted_comps, axis=0)

        cmp_man_minus = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # projected_comps shape: (n_categories-1, sub_obs_dim)
        projected_comps = cmp_man_minus.map(
            self.int_obs_emb.project,
            weighted_comps[1:],
        )
        # int_means shape: (sub_obs_dim, n_categories-1)
        int_means = self.int_man.from_columns(projected_comps)

        return self.join_coords(obs_means, int_means, weights)

    # Overrides

    @property
    @override
    def int_obs_emb(self) -> LinearEmbedding[IntObservable, Observable]:
        return self._int_obs_emb

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self.n_categories)

    @property
    @override
    def int_pst_emb(self) -> IdentityEmbedding[Categorical]:
        return IdentityEmbedding(self.lat_man)

    @property
    @override
    def int_man(self) -> RectangularMap[Categorical, IntObservable]:
        return RectangularMap(Rectangular(), self.lat_man, self.int_obs_emb.sub_man)

    # Methods

    def split_natural_mixture(
        self,
        natural_params: Array,
    ) -> tuple[Array, Array]:  # (Natural[Product[Observable]], Natural[Categorical])
        """Split a mixture model in natural coordinates into components and prior.

        In natural coordinates, embeddings are the correct operation because they preserve
        the log-density form of exponential families. The interaction parameters in
        $\\text{IntObservable}$ space are embedded into $\\text{Observable}$ space by
        translating the observable bias: $\\theta_k = \\text{embed}(\\theta_X, \\delta_k)$
        ensures that $\\log p(x|z=k) = \\theta_k \\cdot s(x) - \\psi(\\theta_k) + \\text{const}$.

        This operation is invertible for general $\\text{IntObservable} \\subsetneq \\text{Observable}$
        because zero-padding in natural space preserves the exponential family structure.
        """

        lkl_params, prr_params = self.split_conjugated(natural_params)
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)

        # into_cols shape: (n_categories - 1, sub_obs_dim)
        int_cols = self.int_man.to_columns(int_mat)

        def translate_col(
            col: Array,
        ) -> Array:
            return self.int_obs_emb.translate(obs_bias, col)

        # translated shape: (n_categories - 1, obs_dim)
        translated = self.int_man.col_man.map(translate_col, int_cols)
        # components shape: (n_categories, obs_dim)
        components = jnp.vstack([obs_bias[None, :], translated])

        return components, prr_params

    def observable_mean_covariance(
        self,
        natural_params: Array,
    ) -> tuple[Array, Array]:
        """Compute mean and covariance of the observable variables."""
        if not isinstance(self.obs_man, StatisticalMoments):
            raise TypeError(
                f"Observable manifold {type(self.obs_man)} does not support statistical moment computation"
            )

        comp_params, cat_params = self.split_natural_mixture(natural_params)
        weights = self.lat_man.to_probs(self.lat_man.to_mean(cat_params))

        # Use statistical_mean/covariance for component statistics
        cmp_means = self.cmp_man.map(self.obs_man.statistical_mean, comp_params)
        cmp_covs = self.cmp_man.map(self.obs_man.statistical_covariance, comp_params)

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
        lkl_params: Array,
    ) -> Array:
        """Compute conjugation parameters for categorical mixture. In particular,

        $$\\rho_k = \\psi(\\theta_X + \\theta_{XZ,k}) - \\psi(\\theta_X)$$
        """
        # Compute base term from observable bias
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)
        rho_0 = self.obs_man.log_partition_function(obs_bias)

        # int_comps shape: (n_categories - 1, sub_obs_dim)
        int_comps = self.int_man.to_columns(int_mat)

        def compute_rho(comp_params: Array) -> Array:
            adjusted_obs = self.int_obs_emb.translate(obs_bias, comp_params)
            return self.obs_man.log_partition_function(adjusted_obs) - rho_0

        # rho_z shape: (n_categories - 1,)
        return self.int_man.col_man.map(compute_rho, int_comps)


class CompleteMixture[Observable: Differentiable](
    Mixture[Observable, Observable],
):
    """Mixture models where the complete observable manifold is mixed."""

    # Constructor

    def __init__(self, obs_man: Observable, n_categories: int):
        super().__init__(n_categories, IdentityEmbedding(obs_man))

    def split_mean_mixture(
        self,
        means: Array,
    ) -> tuple[Array, Array]:
        """Split a mixture model in mean coordinates into components and weights.

        **Constraint:** This method is restricted to $\\text{Observable} = \\text{IntObservable}$.
        Since `join_mean_mixture` projects component statistics to the interaction submanifold,
        information outside that submanifold is irreversibly lost. Only when the full Observable
        space is used for interactions can the decomposition perfectly invert the join operation.
        """
        obs_means, int_means, cat_means = self.split_coords(means)
        probs = self.lat_man.to_probs(cat_means)  # shape: (n_categories,)

        # Get interaction columns - shape: (n_categories-1, obs_dim)
        int_cols = self.int_man.to_columns(int_means)

        # Compute first component
        # Sum interaction columns - shape: (obs_dim,)
        sum_interactions = jnp.sum(int_cols, axis=0)
        first_comp = (obs_means - sum_interactions) / probs[0]

        # Scale remaining components by their probabilities
        # shape: (n_categories-1, obs_dim)
        other_comps = int_cols / probs[1:, None]

        # Somehow force the above calls to be evaluated before the next line
        jax.block_until_ready(first_comp)
        jax.block_until_ready(other_comps)

        # Combine components - shape: (n_categories, obs_dim)
        components = jnp.vstack([first_comp[None, :], other_comps])

        return components, cat_means

    def join_natural_mixture(
        self,
        components: Array,
        prior: Array,
    ) -> Array:
        """Create a mixture model in natural coordinates from components and prior.

        **Constraint:** This method is restricted to $\\text{Observable} = \\text{IntObservable}$.
        Component differences $\\theta_k - \\theta_0$ are computed in Observable space and stored
        directly. For general submanifolds, these differences would need to be projected to
        $\\text{IntObservable}$ space, but projection in natural coordinates does not preserve
        the exponential family structure (would corrupt the log-density).

        Algorithm: Uses the first component as an anchor, storing differences in the
        interaction matrix: $\\Theta_{XZ,k} = \\theta_k - \\theta_0$.
        """
        # Get anchor (first component) - shape: (obs_dim,)
        obs_bias = components[0]

        def to_interaction(
            comp: Array,
        ) -> Array:
            return comp - obs_bias

        cmp_man_minus = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # projected_comps shape: (n_categories-1, obs_dim)
        projected_comps = cmp_man_minus.map(to_interaction, components[1:])

        # int_mat shape: (sub_obs_dim, n_categories-1)
        int_mat = self.int_man.from_columns(projected_comps)
        lkl_params = self.lkl_fun_man.join_coords(obs_bias, int_mat)

        return self.join_conjugated(lkl_params, prior)


class AnalyticMixture[Observable: Analytic](
    CompleteMixture[Observable],
    AnalyticConjugated[Observable, Observable, Categorical, Categorical],
):
    """Mixture model with analytically tractable components. Amongst other things, this enables closed-form implementation of expectation-maximization."""

    # Overrides

    @override
    def to_natural_likelihood(
        self,
        means: Array,
    ) -> Array:
        """Map mean harmonium parameters to natural likelihood parameters.

        **Constraint:** This method requires $\\text{Observable} = \\text{IntObservable}$. This is
        necessary because converting from mean to natural coordinates requires decomposing the
        mixture into individual component parameters. In the general case where interactions are
        restricted to a submanifold, information outside that submanifold is irreversibly lost
        during `join_mean_mixture`, making the decomposition impossible.

        Algorithm:
        1. Recover component means using `split_mean_mixture` (requires full Observable recovery)
        2. Convert each component mean parameter to natural parameters
        3. Compute differences relative to first component (interaction matrix)
        """
        # Get component means - shape: (n_categories, obs_dim)
        comp_means, _ = self.split_mean_mixture(means)

        # Convert each component to natural parameters
        def to_natural(mean: Array) -> Array:
            return self.obs_man.to_natural(mean)

        # nat_comps shape: (n_categories, obs_dim)
        nat_comps = self.cmp_man.map(to_natural, comp_means)

        # Get anchor (first component) - shape: (obs_dim,)
        obs_bias = nat_comps[0]

        def to_interaction(
            nat: Array,
        ) -> Array:
            return nat - obs_bias

        # Convert remaining components to interactions
        cmp_man1 = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # int_cols shape: (n_categories-1, obs_dim)
        int_cols = cmp_man1.map(to_interaction, nat_comps[1:])

        # int_mat shape: (obs_dim, n_categories-1)
        int_mat = self.int_man.from_columns(int_cols)

        return self.lkl_fun_man.join_coords(obs_bias, int_mat)
