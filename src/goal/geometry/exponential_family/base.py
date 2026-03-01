"""Exponential family hierarchy: ExponentialFamily, Gibbs, Generative, Differentiable, Analytic.

Each level adds capabilities --- sufficient statistics, Gibbs sampling, i.i.d. sampling, log-partition function, or negative entropy --- that unlock progressively more powerful inference algorithms.

Variable names encode the coordinate system throughout: ``params`` for natural parameters (with prefixed variants like ``obs_params`` for slices), ``means`` for mean parameters, and ``coords`` for coordinate-system-agnostic arrays in the manifold layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Manifold
from ..manifold.util import batched_mean

### Exponential Families ###


class ExponentialFamily(Manifold, ABC):
    """A statistical manifold whose points are probability distributions in an exponential family.

    Subclasses define the sufficient statistic $\\mathbf{s}(x)$ and base measure $\\mu(x)$; higher levels of the hierarchy add the normalizing constant and its dual.

    Mathematically, an exponential family is a set of distributions whose densities share the form $p(x; \\theta) \\propto \\mu(x)\\exp(\\theta \\cdot \\mathbf{s}(x))$, where $\\theta \\in \\mathbb{R}^n$ are the natural parameters, $\\mathbf{s}(x)$ is the sufficient statistic --- a fixed mapping from data to $\\mathbb{R}^n$ that captures all information the data carries about $\\theta$ --- and $\\mu(x)$ is the base measure, a fixed reference density independent of $\\theta$.
    """

    # Contract

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimension of the data space."""

    @abstractmethod
    def sufficient_statistic(self, x: Array) -> Array:
        """Compute the sufficient statistic $\\mathbf{s}(x)$ of an observation."""

    @abstractmethod
    def log_base_measure(self, x: Array) -> Array:
        """Compute $\\log \\mu(x)$ for an observation."""

    # Methods

    def average_sufficient_statistic(self, xs: Array, batch_size: int = 256) -> Array:
        """Average sufficient statistics over a batch of observations."""
        return batched_mean(self.sufficient_statistic, xs, batch_size)

    def check_natural_parameters(self, params: Array) -> Array:
        """Check if the given natural parameters are valid (all finite)."""
        return jnp.all(jnp.isfinite(params)).astype(jnp.int32)

    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Generate random natural parameters from a Gaussian perturbation."""
        return jax.random.normal(key, shape=(self.dim,)) * shape + location

    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,  # pyright: ignore[reportUnusedParameter]
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Generate random natural parameters, optionally informed by data.

        Default: ignores the sample. ``Analytic`` overrides this to use average sufficient statistics.
        """
        return self.initialize(key, location, shape)


class Gibbs(ExponentialFamily, ABC):
    """Adds Gibbs sampling to an exponential family, enabling MCMC-based inference for models without direct i.i.d. sampling."""

    # Contract

    @abstractmethod
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        """Perform one Gibbs sampling step given natural parameters and a current state."""

    # Methods

    def gibbs_chain(
        self,
        key: Array,
        params: Array,
        initial_state: Array,
        k: int,
    ) -> Array:
        """Run ``k`` Gibbs sampling steps given natural parameters, returning the final state."""

        def step(state: Array, step_idx: Array) -> tuple[Array, None]:
            subkey = jax.random.fold_in(key, step_idx)
            return self.gibbs_step(subkey, params, state), None

        final_state, _ = jax.lax.scan(step, initial_state, jnp.arange(k))
        return final_state


class Generative(Gibbs, ABC):
    """Adds i.i.d. sampling to a Gibbs-capable exponential family, enabling Monte Carlo estimation when closed-form expressions are unavailable."""

    # Contract

    @abstractmethod
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Draw ``n`` samples from the distribution with the given natural parameters."""

    # Overrides

    @override
    def gibbs_step(
        self,
        key: Array,
        params: Array,
        state: Array,
    ) -> Array:
        """Perform one Gibbs sampling step given natural parameters and a current state.

        Default: samples independently, ignoring state. Override for models with efficient conditional sampling.
        """
        return self.sample(key, params, n=1)[0]

    # Methods

    def stochastic_to_mean(self, key: Array, params: Array, n: int) -> Array:
        """Estimate average sufficient statistics by sampling from the given natural parameters."""
        samples = self.sample(key, params, n)
        return self.average_sufficient_statistic(samples)


class Differentiable(Generative, ABC):
    """Adds an analytic log-partition function, enabling exact density evaluation and gradient-based optimization.

    Mathematically, the log-partition function $\\psi(\\theta) = \\log \\int \\mu(x)\\exp(\\theta \\cdot \\mathbf{s}(x))\\,dx$ normalizes the density. Its gradient defines the mean parameters $\\eta = \\nabla\\psi(\\theta) = \\mathbb{E}_{p(x;\\theta)}[\\mathbf{s}(x)]$, providing a dual coordinate system on the manifold.
    """

    # Contract

    @abstractmethod
    def log_partition_function(self, params: Array) -> Array:
        """Compute the log-partition function $\\psi$ at the given natural parameters."""

    # Methods

    def to_mean(self, params: Array) -> Array:
        """Convert natural parameters to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
        return jax.grad(self.log_partition_function)(params)

    def log_density(self, params: Array, x: Array) -> Array:
        """Evaluate log-density at observation $x$ under the given natural parameters.

        Computes $\\log p(x;\\theta) = \\theta \\cdot \\mathbf{s}(x) + \\log \\mu(x) - \\psi(\\theta)$.
        """
        suff_stats = self.sufficient_statistic(x)
        return (
            jnp.dot(params, suff_stats)
            + self.log_base_measure(x)
            - self.log_partition_function(params)
        )

    def density(self, params: Array, x: Array) -> Array:
        """Evaluate density at observation $x$ under the given natural parameters."""
        return jnp.exp(self.log_density(params, x))

    def relative_entropy(self, p_params: Array, q_params: Array) -> Array:
        """Compute KL divergence $D(p \\| q)$ between two distributions given their natural parameters.

        Uses the Bregman divergence form: $D(p \\| q) = \\psi(\\theta_q) - \\psi(\\theta_p) + \\eta_p \\cdot (\\theta_p - \\theta_q)$.
        """
        p_means = self.to_mean(p_params)
        psi_p = self.log_partition_function(p_params)
        psi_q = self.log_partition_function(q_params)
        return psi_q - psi_p + jnp.dot(p_means, p_params - q_params)

    def average_log_density(
        self, params: Array, xs: Array, batch_size: int = 2048
    ) -> Array:
        """Average log-density over a batch of observations under the given natural parameters."""

        def _log_density(x: Array) -> Array:
            return self.log_density(params, x)

        return batched_mean(_log_density, xs, batch_size)


class Analytic(Differentiable, ABC):
    """Adds a closed-form negative entropy $\\phi(\\eta)$, completing the duality between natural and mean coordinates.

    Mathematically, $\\phi(\\eta) = \\sup_{\\theta}\\{\\theta \\cdot \\eta - \\psi(\\theta)\\}$ is the Legendre conjugate of $\\psi$, and $\\theta = \\nabla\\phi(\\eta)$ inverts the natural-to-mean mapping.

    **NB:** This negative entropy is the convex conjugate of the log-partition function and does not include the base measure term. It may differ from the entropy as defined in information theory.
    """

    # Contract

    @abstractmethod
    def negative_entropy(self, means: Array) -> Array:
        """Compute negative entropy $\\phi$ at the given mean parameters."""

    # Overrides

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize natural parameters from noisy average sufficient statistics of the sample."""
        avg_suff_stat = self.average_sufficient_statistic(sample)
        noise = jax.random.normal(key, shape=(self.dim,)) * shape + location
        noisy_means = avg_suff_stat + noise
        return self.to_natural(noisy_means)

    # Methods

    def to_natural(self, means: Array) -> Array:
        """Convert mean parameters to natural parameters via $\\theta = \\nabla\\phi(\\eta)$."""
        return jax.grad(self.negative_entropy)(means)
