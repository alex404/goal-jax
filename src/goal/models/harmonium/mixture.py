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
    EmbeddedMap,
    IdentityEmbedding,
    LinearEmbedding,
    Manifold,
    Product,
    Rectangular,
    StatisticalMoments,
    SymmetricConjugated,
)
from ..base.categorical import (
    Categorical,
)


@dataclass(frozen=True)
class Mixture[Observable: Differentiable](
    SymmetricConjugated[Observable, Categorical],
    DifferentiableConjugated[Observable, Categorical, Categorical],
):
    """Mixture models with exponential family observations.

    A mixture model represents a weighted combination of component distributions:

    $$p(x) = \\sum_{k=1}^K \\pi_k p(x|\\theta_k)$$

    where $\\pi_k$ are mixture weights summing to 1, and $p(x|\\theta_k)$ are component
    distributions from the same exponential family.

    In the harmonium framework, mixture models are implemented using:
        - A categorical latent variable $z$ representing component assignment
        - An observable distribution family for component distributions
        - An interaction matrix between $z$ and $x$ that contain component specific parameters

    The interaction matrix structure is fixed: Rectangular matrix representation with
    identity domain embedding. Only the observable embedding varies.
    """

    # Fields

    n_categories: int
    """Number of mixture components."""

    obs_emb: LinearEmbedding[Manifold, Observable]
    """Observable embedding - determines the interaction submanifold structure."""

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

        **Technical note:** When using restricted submanifolds (non-identity embeddings), component
        statistics outside the interaction submanifold are irreversibly lost (aggregated into
        observable biases). This is mathematically correct for mean coordinates but makes
        the operation non-invertible without additional information.

        Args:
            components: Flat 1D array of component parameters (shape ``[n_categories * obs_dim]``)
            weights: Categorical weights
        """
        probs = self.pst_man.to_probs(weights)

        comps_2d = self.cmp_man.to_2d(components)
        weighted_comps = comps_2d * probs[:, None]

        obs_means = jnp.sum(weighted_comps, axis=0)

        # Project components (excluding first) to interaction subspace
        projected_comps = jax.vmap(self.int_man.cod_emb.project)(weighted_comps[1:])
        # [n_categories-1, sub_obs_dim]

        # Transpose and convert to int_man storage format
        int_means = projected_comps.T.ravel()

        return self.join_coords(obs_means, int_means, weights)

    # Overrides

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self.n_categories)

    @property
    @override
    def int_man(self) -> EmbeddedMap[Categorical, Observable]:
        """Construct interaction matrix from observable embedding.

        Structure is fixed: Rectangular matrix with identity domain embedding.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.lat_man),
            self.obs_emb,
        )

    # Methods

    def split_natural_mixture(
        self,
        natural_params: Array,
    ) -> tuple[Array, Array]:
        """Split a mixture model in natural coordinates into components and prior.

        In natural coordinates, embeddings are the correct operation because they preserve
        the log-density form of exponential families. The interaction parameters in
        $\\text{IntObservable}$ space are embedded into $\\text{Observable}$ space by
        translating the observable bias: $\\theta_k = \\text{embed}(\\theta_X, \\delta_k)$
        ensures that $\\log p(x|z=k) = \\theta_k \\cdot s(x) - \\psi(\\theta_k) + \\text{const}$.

        This operation is invertible for general $\\text{IntObservable} \\subsetneq \\text{Observable}$
        because zero-padding in natural space preserves the exponential family structure.

        Returns:
            Tuple of (components, prior) where components is a flat 1D array of shape
            ``[n_categories * obs_dim]`` representing parameters on ``cmp_man``.
        """

        lkl_params, prr_params = self.split_conjugated(natural_params)
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)

        # Convert to 2D matrix and transpose to get columns as rows
        int_cols = self.int_man.to_matrix(int_mat).T  # [n_categories-1, sub_obs_dim]

        # Translate each column from subspace to full observable space
        def translate_col(col: Array) -> Array:
            return self.int_man.cod_emb.translate(obs_bias, col)

        translated = jax.vmap(translate_col)(int_cols)

        # Stack with first component and flatten to match cmp_man convention
        components = jnp.vstack([obs_bias[None, :], translated]).ravel()

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

        # Convert to 2D matrix and transpose to get columns as rows
        int_comps = self.int_man.to_matrix(int_mat).T  # [n_categories-1, sub_obs_dim]

        def compute_rho(comp_params: Array) -> Array:
            adjusted_obs = self.int_man.cod_emb.translate(obs_bias, comp_params)
            return self.obs_man.log_partition_function(adjusted_obs) - rho_0

        return jax.vmap(compute_rho)(int_comps)  # [n_categories-1]


class CompleteMixture[Observable: Differentiable](
    Mixture[Observable],
):
    """Mixture models where the complete observable manifold is mixed."""

    # Constructor

    def __init__(self, obs_man: Observable, n_categories: int):
        # Use identity observable embedding for complete mixture
        obs_emb = IdentityEmbedding(obs_man)
        super().__init__(n_categories, obs_emb)

    def split_mean_mixture(
        self,
        means: Array,
    ) -> tuple[Array, Array]:
        """Split a mixture model in mean coordinates into components and weights.

        **Constraint:** This method is restricted to $\\text{Observable} = \\text{IntObservable}$.
        Since `join_mean_mixture` projects component statistics to the interaction submanifold,
        information outside that submanifold is irreversibly lost. Only when the full Observable
        space is used for interactions can the decomposition perfectly invert the join operation.

        Returns:
            Tuple of (components, weights) where components is a flat 1D array of shape
            ``[n_categories * obs_dim]`` representing parameters on ``cmp_man``.
        """
        obs_means, int_means, cat_means = self.split_coords(means)
        probs = self.lat_man.to_probs(cat_means)  # shape: (n_categories,)

        # Convert to 2D matrix and transpose to get columns as rows [n_categories-1, obs_dim]
        int_dense = self.int_man.to_matrix(int_means)  # [obs_dim, n_categories-1]
        int_cols = int_dense.T  # [n_categories-1, obs_dim]

        # Compute first component
        # Sum interaction columns - shape: (obs_dim,)
        sum_interactions = jnp.sum(int_cols, axis=0)
        first_comp = (obs_means - sum_interactions) / probs[0]

        # Scale remaining components by their probabilities
        # shape: (n_categories-1, obs_dim)
        other_comps = int_cols / probs[1:, None]

        # Combine components and flatten to match cmp_man convention
        components = jnp.vstack([first_comp[None, :], other_comps]).ravel()

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

        Args:
            components: Flat 1D array of component parameters (shape ``[n_categories * obs_dim]``)
            prior: Prior parameters for categorical distribution
        """
        # Get anchor (first component) - shape: (obs_dim,)
        obs_bias = self.cmp_man.get_replicate(components, 0)

        def to_interaction(comp: Array) -> Array:
            return comp - obs_bias

        # Get remaining components (flat 1D)
        remaining_start = self.obs_man.dim
        components_rest = components[remaining_start:]

        cmp_man_minus = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # projected_comps shape: (n_categories-1, obs_dim)
        projected_comps = cmp_man_minus.map(to_interaction, components_rest)

        # Transpose to [obs_dim, n_categories-1] and convert to int_man storage
        int_mat = self.int_man.rep.from_matrix(projected_comps.T)
        lkl_params = self.lkl_fun_man.join_coords(obs_bias, int_mat)

        return self.join_conjugated(lkl_params, prior)


class AnalyticMixture[Observable: Analytic](
    CompleteMixture[Observable],
    AnalyticConjugated[Observable, Categorical],
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
        # Get component means (flat 1D)
        comp_means, _ = self.split_mean_mixture(means)

        # Convert each component to natural parameters (flat 1D)
        def to_natural(mean: Array) -> Array:
            return self.obs_man.to_natural(mean)

        nat_comps = self.cmp_man.map(to_natural, comp_means, flatten=True)

        # Get anchor (first component) - shape: (obs_dim,)
        obs_bias = self.cmp_man.get_replicate(nat_comps, 0)

        def to_interaction(nat: Array) -> Array:
            return nat - obs_bias

        # Convert remaining components to interactions
        remaining_start = self.obs_man.dim
        nat_comps_rest = nat_comps[remaining_start:]

        cmp_man1 = replace(self.cmp_man, n_reps=self.n_categories - 1)
        # int_cols shape: (n_categories-1, obs_dim)
        int_cols = cmp_man1.map(to_interaction, nat_comps_rest)

        # Transpose to [obs_dim, n_categories-1] and convert to int_man storage
        int_mat = self.int_man.rep.from_matrix(int_cols.T)

        return self.lkl_fun_man.join_coords(obs_bias, int_mat)
