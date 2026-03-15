"""Harmonium models: product exponential families over observable and latent variables coupled through an interaction matrix.

Natural parameters of a harmonium are the concatenation ``[obs_params, int_params, lat_params]`` of observable biases, interaction matrix, and latent biases. The conjugated subclasses add structure that decomposes the joint into a likelihood and a prior, enabling exact density evaluation and EM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Triple
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from ..manifold.linear import AffineMap, LinearMap
from ..manifold.util import batched_mean
from .base import (
    Analytic,
    Differentiable,
    Generative,
    Gibbs,
)


@dataclass(frozen=True)
class Harmonium[
    Observable: Gibbs,
    Posterior: Gibbs,
](
    Gibbs,
    Triple[Observable, LinearMap[Posterior, Observable], Posterior],
    ABC,
):
    """A product exponential family over observable $x$ and latent $z$ variables coupled through an interaction matrix.

    Mathematically, the joint log-density is $\\log p(x,z) = \\theta_X \\cdot \\mathbf s_X(x) + \\theta_Z \\cdot \\mathbf s_Z(z) + \\mathbf s_X(x) \\cdot \\Theta_{XZ} \\cdot \\mathbf s_Z(z) - \\psi(\\theta)$, where $\\theta_X$, $\\theta_Z$ are observable and latent biases, and $\\Theta_{XZ}$ is the interaction matrix.
    """

    # Contract

    @property
    @abstractmethod
    def int_man(self) -> LinearMap[Posterior, Observable]:
        """Manifold of the interaction matrix $\\Theta_{XZ}$."""

    # Methods

    @property
    def obs_man(self) -> Observable:
        """Manifold of observable biases."""
        return self.int_man.cod_man

    @property
    def pst_man(self) -> Posterior:
        """Manifold of posterior specific latent biases."""
        return self.int_man.dom_man

    @property
    def lkl_fun_man(self) -> AffineMap[Posterior, Observable]:
        """Manifold of likelihood distributions $p(x \\mid z)$."""
        return AffineMap(self.int_man, self.int_man.dom_man)

    @property
    def pst_fun_man(self) -> AffineMap[Observable, Posterior]:
        """Manifold of conditional posterior distributions $p(z \\mid x)$."""
        return AffineMap(self.int_man.trn_man, self.int_man.cod_man)

    def likelihood_function(self, params: Array) -> Array:
        """Extract the likelihood affine map $\\eta \\mapsto \\theta_X + \\Theta_{XZ} \\cdot \\eta$ from the given natural parameters."""
        obs_params, int_params, _ = self.split_coords(params)
        return self.lkl_fun_man.join_coords(obs_params, int_params)

    def posterior_function(self, params: Array) -> Array:
        """Extract the posterior affine map $\\eta \\mapsto \\theta_Z + \\Theta_{XZ}^\\top \\cdot \\eta$ from the given natural parameters."""
        _, int_params, lat_params = self.split_coords(params)
        int_mat_t = self.int_man.transpose(int_params)
        return self.pst_fun_man.join_coords(lat_params, int_mat_t)

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Evaluate the likelihood natural parameters $p(x \\mid z)$ at a given latent state $z$."""
        mz = self.pst_man.sufficient_statistic(z)
        return self.lkl_fun_man(self.likelihood_function(params), mz)

    def posterior_at(self, params: Array, x: Array) -> Array:
        """Evaluate the posterior natural parameters $p(z \\mid x)$ at a given observation $x$."""
        mx = self.obs_man.sufficient_statistic(x)
        return self.pst_fun_man(self.posterior_function(params), mx)

    # Overrides

    @property
    @override
    def fst_man(self) -> Observable:
        return self.obs_man

    @property
    @override
    def snd_man(self) -> LinearMap[Posterior, Observable]:
        return self.int_man

    @property
    @override
    def trd_man(self) -> Posterior:
        return self.pst_man

    @property
    @override
    def data_dim(self) -> int:
        """Total dimension of data points."""
        return self.obs_man.data_dim + self.pst_man.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Compute sufficient statistics of a joint observation $[x, z]$."""
        obs_x = x[..., : self.obs_man.data_dim]
        lat_x = x[..., self.obs_man.data_dim :]

        obs_stats = self.obs_man.sufficient_statistic(obs_x)
        lat_stats = self.pst_man.sufficient_statistic(lat_x)

        int_stats = self.int_man.outer_product(obs_stats, lat_stats)

        return self.join_coords(obs_stats, int_stats, lat_stats)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Compute log base measure of a joint observation $[x, z]$."""
        obs_x = x[..., : self.obs_man.data_dim]
        lat_x = x[..., self.obs_man.data_dim :]

        return self.obs_man.log_base_measure(obs_x) + self.pst_man.log_base_measure(
            lat_x
        )

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize harmonium natural parameters with biases from component initialization and interaction matrix scaled by ``shape / sqrt(int_dim)``."""
        keys = jax.random.split(key, 3)

        obs_params = self.obs_man.initialize(keys[0], location, shape)
        lat_params = self.pst_man.initialize(keys[1], location, shape)

        scaling = shape / jnp.sqrt(self.int_man.dim)
        int_params = scaling * jax.random.normal(keys[2], shape=[self.int_man.dim])

        return self.join_coords(obs_params, int_params, lat_params)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize harmonium natural parameters, using sample data for observable biases."""
        keys = jax.random.split(key, 3)

        obs_params = self.obs_man.initialize_from_sample(
            keys[0], sample, location, shape
        )
        lat_params = self.pst_man.initialize(keys[1], location, shape)

        scaling = shape / jnp.sqrt(self.int_man.dim)
        int_params = scaling * jax.random.normal(keys[2], shape=[self.int_man.dim])

        return self.join_coords(obs_params, int_params, lat_params)

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        """One Gibbs sweep: sample $x \\sim p(x \\mid z)$, then $z \\sim p(z \\mid x)$."""
        key1, key2 = jax.random.split(key)

        obs_dim = self.obs_man.data_dim
        x, z = state[:obs_dim], state[obs_dim:]

        lkl_params = self.likelihood_at(params, z)
        x_new = self.obs_man.gibbs_step(key1, lkl_params, x)

        post_params = self.posterior_at(params, x_new)
        z_new = self.pst_man.gibbs_step(key2, post_params, z)

        result = jnp.concatenate([x_new, z_new])
        return result.astype(state.dtype)

    def contrastive_divergence_step(
        self,
        key: Array,
        params: Array,
        x: Array,
        k: int = 1,
    ) -> Array:
        """Compute CD-k gradient contribution for a single observation at the given natural parameters.

        Positive phase: sample $z_0$ from $p(z \\mid x)$ with $x$ clamped. Negative phase: run $k$ joint Gibbs steps from $(x, z_0)$. Returns negative minus positive sufficient statistics, approximating $\\mathbb{E}_{\\text{model}}[\\mathbf{s}] - \\mathbb{E}_{\\text{data}}[\\mathbf{s}]$.
        """
        key1, key2 = jax.random.split(key)

        # Positive phase: sample z from posterior with x clamped
        post_params = self.posterior_at(params, x)
        z_0 = self.pst_man.gibbs_step(
            key1, post_params, jnp.zeros(self.pst_man.data_dim)
        )
        positive_state = jnp.concatenate([x, z_0])
        positive_stats = self.sufficient_statistic(positive_state)

        # Negative phase: k joint Gibbs steps from (x, z_0)
        negative_state = self.gibbs_chain(key2, params, positive_state, k)
        negative_stats = self.sufficient_statistic(negative_state)

        return negative_stats - positive_stats

    def mean_contrastive_divergence_gradient(
        self,
        key: Array,
        params: Array,
        xs: Array,
        k: int = 1,
    ) -> Array:
        """Compute average CD-k gradient over a batch of observations at the given natural parameters."""
        n = xs.shape[0]
        keys = jax.random.split(key, n)

        cd_grads = jax.vmap(
            self.contrastive_divergence_step, in_axes=(0, None, 0, None)
        )(keys, params, xs, k)

        return jnp.mean(cd_grads, axis=0)


class Conjugated[
    Observable: Differentiable,
    Posterior: Gibbs,
    Prior: Generative,
](
    Harmonium[Observable, Posterior],
    Generative,
    ABC,
):
    """A harmonium whose prior $p(z)$ belongs to the same exponential family as the posterior $p(z \\mid x)$, enabling exact computation of the prior via conjugation parameters $\\rho$."""

    # Contract

    @property
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Posterior, Prior]:
        """Embedding of the posterior latent submanifold into the prior latent manifold."""

    @abstractmethod
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters $\\rho$ from the given likelihood natural parameters."""

    # Methods

    @property
    def prr_man(self) -> Prior:
        """Manifold of the prior distribution."""
        return self.pst_prr_emb.amb_man

    def prior(self, params: Array) -> Array:
        """Compute prior natural parameters $p(z)$ from the given harmonium natural parameters."""
        obs_params, int_params, lat_params = self.split_coords(params)
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)
        rho = self.conjugation_parameters(lkl_params)
        return self.pst_prr_emb.translate(rho, lat_params)

    def split_conjugated(self, params: Array) -> tuple[Array, Array]:
        """Split harmonium natural parameters into likelihood and prior natural parameters."""
        lkl_params = self.likelihood_function(params)
        return lkl_params, self.prior(params)

    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract the variables needed to condition the likelihood from a prior sample.

        Returns the full sample by default. Hierarchical models override to extract only the immediate child latent (e.g., $y$ from a joint $yz$ sample).
        """
        return prr_sample

    def observable_sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from the observable marginal $p(x)$ by discarding latent components."""
        xzs = self.sample(key, params, n)
        return xzs[:, : self.obs_man.data_dim]

    # Overrides

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from the joint by first sampling $z \\sim p(z)$, then $x \\sim p(x \\mid z)$."""
        key1, key2 = jax.random.split(key)

        nat_prior = self.prior(params)
        z_sample = self.prr_man.sample(key1, nat_prior, n)
        z0_sample = self.extract_likelihood_input(z_sample)

        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(params, z0_sample)
        x_keys = jax.random.split(key2, n)
        x_sample = jax.vmap(self.obs_man.sample, in_axes=(0, 0, None))(
            x_keys, x_params, 1
        ).reshape((n, -1))

        return jnp.concatenate([x_sample, z_sample], axis=-1)


class DifferentiableConjugated[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
](
    Conjugated[Observable, Posterior, Prior],
    Differentiable,
    ABC,
):
    """A conjugated harmonium with an analytical log-partition function, enabling exact density evaluation and gradient-based optimization."""

    # Overrides

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Compute $\\psi(\\theta) = \\psi_Z(\\theta_Z + \\rho) + \\psi_X(\\theta_X)$ at the given natural parameters."""
        obs_params, int_params, lat_params = self.split_coords(params)
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)

        chi = self.obs_man.log_partition_function(obs_params)
        rho = self.conjugation_parameters(lkl_params)
        adjusted_lat = self.pst_prr_emb.translate(rho, lat_params)

        return self.prr_man.log_partition_function(adjusted_lat) + chi

    # Methods

    def log_observable_density(self, params: Array, x: Array) -> Array:
        """Compute log marginal density $\\log p(x)$ at the given natural parameters by integrating out the latent variable analytically."""
        obs_params, _, _ = self.split_coords(params)

        chi = self.obs_man.log_partition_function(obs_params)
        obs_stats = self.obs_man.sufficient_statistic(x)
        prr = self.prior(params)

        log_density = jnp.dot(obs_params, obs_stats)
        log_density += self.pst_man.log_partition_function(self.posterior_at(params, x))
        log_density -= self.prr_man.log_partition_function(prr) + chi

        return log_density + self.obs_man.log_base_measure(x)

    def observable_density(self, params: Array, x: Array) -> Array:
        """Compute marginal density $p(x)$ at the given natural parameters."""
        return jnp.exp(self.log_observable_density(params, x))

    def average_log_observable_density(
        self, params: Array, xs: Array, batch_size: int = 2048
    ) -> Array:
        """Compute average $\\log p(x)$ over a batch of observations at the given natural parameters."""

        def _log_density(x: Array) -> Array:
            return self.log_observable_density(params, x)

        return batched_mean(_log_density, xs, batch_size)

    def posterior_statistics(self, params: Array, x: Array) -> Array:
        """Compute expected sufficient statistics $(\\mathbf s_X(x),\\, \\mathbf s_X(x) \\otimes \\mathbb E[\\mathbf s_Z \\mid x],\\, \\mathbb E[\\mathbf s_Z \\mid x])$ in mean coordinates for a single observation, at the given natural parameters."""
        obs_stats = self.obs_man.sufficient_statistic(x)

        post_map = self.posterior_function(params)
        lat_params = self.pst_fun_man(post_map, obs_stats)
        lat_means = self.pst_man.to_mean(lat_params)

        int_means = self.int_man.outer_product(obs_stats, lat_means)
        return self.join_coords(obs_stats, int_means, lat_means)

    def mean_posterior_statistics(
        self,
        params: Array,
        xs: Array,
        batch_size: int = 256,
    ) -> Array:
        """Compute average expected sufficient statistics over a batch of observations at the given natural parameters."""

        def infer_missing_expectations(x: Array) -> Array:
            return self.posterior_statistics(params, x)

        return batched_mean(infer_missing_expectations, xs, batch_size)


class SymmetricConjugated[
    Observable: Differentiable,
    Latent: Differentiable,
](
    DifferentiableConjugated[Observable, Latent, Latent],
    ABC,
):
    """A conjugated harmonium where the posterior and prior share the same latent manifold (``pst_man == prr_man``)."""

    # Contract

    @property
    @abstractmethod
    def lat_man(self) -> Latent:
        """Manifold of latent biases."""

    # Overrides

    @property
    @override
    def pst_prr_emb(self) -> LinearEmbedding[Latent, Latent]:
        return IdentityEmbedding(self.lat_man)

    @property
    @override
    def pst_man(self) -> Latent:
        return self.lat_man

    @property
    @override
    def prr_man(self) -> Latent:
        return self.lat_man

    # Methods

    def join_conjugated(self, lkl_params: Array, prior_params: Array) -> Array:
        """Join likelihood and prior natural parameters into harmonium natural parameters."""
        rho = self.conjugation_parameters(lkl_params)
        lat_params = prior_params - rho
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)
        return self.join_coords(obs_params, int_params, lat_params)


class AnalyticConjugated[
    Observable: Differentiable,
    Latent: Analytic,
](
    SymmetricConjugated[Observable, Latent],
    Analytic,
    ABC,
):
    """A symmetric conjugated harmonium with analytically tractable negative entropy, enabling closed-form KL divergence and expectation-maximization."""

    # Contract

    @abstractmethod
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert harmonium mean parameters to likelihood natural parameters."""

    # Overrides

    @override
    def to_natural(self, means: Array) -> Array:
        """Convert harmonium mean parameters to natural parameters."""
        lkl_params = self.to_natural_likelihood(means)
        mean_lat = self.split_coords(means)[2]
        nat_lat = self.pst_man.to_natural(mean_lat)
        return self.join_conjugated(lkl_params, nat_lat)

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Compute negative entropy $\\phi(\\eta) = \\eta \\cdot \\theta - \\psi(\\theta)$ at the given mean parameters."""
        params = self.to_natural(means)

        log_partition = self.log_partition_function(params)
        return jnp.dot(params, means) - log_partition

    # Methods

    def expectation_maximization(self, params: Array, xs: Array) -> Array:
        """Perform one EM iteration: E-step computes expected sufficient statistics, M-step converts to natural parameters."""
        q = self.mean_posterior_statistics(params, xs)
        return self.to_natural(q)
