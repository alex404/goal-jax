"""This module implements the fundamental abstractions for exponential families, providing a hierarchy of classes with increasing capabilities for statistical modeling and inference.

See the package index for mathematical background on exponential families.
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
    """ExponentialFamily defines the basic structure for statistical manifolds, where points represent probability distributions. It provides abstract methods defining the sufficient statistics and base measure. Concrete subclasses implement specific statistical models by defining these components and potentially adding additional capabilities."""

    # Contract

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimension of the data space."""

    @abstractmethod
    def sufficient_statistic(self, x: Array) -> Array:
        """Compute the sufficient statistics of an observation in mean coordinates.

        Args:
            x: Data point

        Returns:
            Sufficient statistics array (mean parameters)
        """

    @abstractmethod
    def log_base_measure(self, x: Array) -> Array:
        """Compute log of base measure $\\mu(x)$."""

    # Templates

    def average_sufficient_statistic(self, xs: Array, batch_size: int = 256) -> Array:
        """Average sufficient statistics of a batch of observations.

        Args:
            xs: Batch of data points
            batch_size: Size of batches for vmapped computation

        Returns:
            Average sufficient statistics (mean parameters)
        """
        return batched_mean(self.sufficient_statistic, xs, batch_size)

    def check_natural_parameters(self, params: Array) -> Array:
        """Check if parameters are valid for this exponential family.

        Args:
            params: Natural parameters

        Returns:
            Scalar indicating validity (all finite)
        """
        return jnp.all(jnp.isfinite(params)).astype(jnp.int32)

    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize natural parameters randomly.

        Args:
            key: JAX random key
            location: Mean of initialization distribution
            shape: Standard deviation of initialization

        Returns:
            Natural parameters array
        """
        return jax.random.normal(key, shape=(self.dim,)) * shape + location

    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,  # pyright: ignore[reportUnusedParameter]
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize based on sample (default: ignore sample).

        Args:
            key: JAX random key
            sample: Data sample to potentially use
            location: Mean of initialization distribution
            shape: Standard deviation of initialization

        Returns:
            Natural parameters array
        """
        return self.initialize(key, location, shape)


class Generative(ExponentialFamily, ABC):
    """Generative extends ExponentialFamily with sampling capabilities, enabling simulation-based methods like Monte Carlo estimation. This allows for working with distributions even when closed-form expressions for statistics are unavailable or intractable.

    In particular, sampling from $p(x; \\theta)$ provides an empirical approach to estimating expectation-based quantities like mean parameters:

    $$\\hat{\\eta} = \\frac{1}{n}\\sum_{i=1}^n \\mathbf{s}(x_i), \\quad x_i \\sim p(x; \\theta).$$
    """

    # Contract

    @abstractmethod
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Generate random samples from the distribution.

        Args:
            key: JAX random key
            params: Natural parameters
            n: Number of samples

        Returns:
            Array of n samples
        """

    # Templates

    def stochastic_to_mean(self, key: Array, params: Array, n: int) -> Array:
        """Estimate mean parameters via Monte Carlo sampling.

        Args:
            key: JAX random key
            params: Natural parameters
            n: Number of samples

        Returns:
            Estimated mean parameters
        """
        samples = self.sample(key, params, n)
        return self.average_sufficient_statistic(samples)


class Differentiable(Generative, ABC):
    """Differentiable adds the ability to compute exact mean parameters from natural parameters, enabling efficient parameter estimation, density evaluation, and data-fitting via gradient descent.

    In particular, the log partition function $\\psi(\\theta)$ is given by:

    $$
    \\psi(\\theta) = \\log \\int_{\\mathcal{X}} \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x))dx
    $$

    Its gradient gives the mean parameters through the dual relationship:

    $$
    \\eta = \\nabla \\psi(\\theta) = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]
    $$
    """

    # Contract

    @abstractmethod
    def log_partition_function(self, params: Array) -> Array:
        """Compute log partition function $\\psi(\\theta)$.

        Args:
            params: Natural parameters

        Returns:
            Log partition function value (scalar)
        """

    # Templates

    def to_mean(self, params: Array) -> Array:
        """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$.

        Args:
            params: Natural parameters

        Returns:
            Mean parameters
        """
        return jax.value_and_grad(self.log_partition_function)(params)[1]

    def log_density(self, params: Array, x: Array) -> Array:
        """Compute log density at x.

        $$
        \\log p(x;\\theta) = \\theta \\cdot \\mathbf s(x) + \\log \\mu(x) - \\psi(\\theta)
        $$

        Args:
            params: Natural parameters
            x: Data point

        Returns:
            Log density (scalar)
        """
        suff_stats = self.sufficient_statistic(x)
        return (
            jnp.dot(params, suff_stats)
            + self.log_base_measure(x)
            - self.log_partition_function(params)
        )

    def density(self, params: Array, x: Array) -> Array:
        """Compute density at x.

        $$
        p(x;\\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
        $$

        Args:
            params: Natural parameters
            x: Data point

        Returns:
            Density (scalar)
        """
        return jnp.exp(self.log_density(params, x))

    def average_log_density(
        self, params: Array, xs: Array, batch_size: int = 2048
    ) -> Array:
        """Compute average log density over a batch of observations.

        Args:
            params: Natural parameters
            xs: Batch of data points
            batch_size: Size of batches for vmapped computation

        Returns:
            Scalar array of the average log density
        """

        def _log_density(x: Array) -> Array:
            return self.log_density(params, x)

        return batched_mean(_log_density, xs, batch_size)


class Analytic(Differentiable, ABC):
    """Analytic provides the highest level of analytical capabilities, supporting bidirectional conversion between natural and mean parameters, and closed-form expression for quantities like entropy and KL divergence calculation.

    In particular, the negative entropy $\\phi(\\eta)$ is the convex conjugate of the log-partition function:

    $$
    \\phi(\\eta) = \\sup_{\\theta} \\{\\theta \\cdot \\eta - \\psi(\\theta)\\}
    $$

    Its gradient gives the natural parameters through the dual relationship:

    $$
    \\theta = \\nabla\\phi(\\eta)
    $$

    This completes the duality between natural and mean coordinates, allowing
    for closed-form solutions to many statistical problems.

    **NB:** This form of negative entropy is the convex conjugate of the log-partition function and does not include the base measure term. It may differ from entropy as traditionally defined in information theory.
    """

    # Contract

    @abstractmethod
    def negative_entropy(self, means: Array) -> Array:
        """Compute negative entropy $\\phi(\\eta)$.

        Args:
            means: Mean parameters

        Returns:
            Negative entropy (scalar)
        """

    # Overrides

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize a model based on the noisy average sufficient statistics.

        Args:
            key: JAX random key
            sample: Data sample
            location: Mean of noise distribution
            shape: Std dev of noise distribution

        Returns:
            Natural parameters array
        """
        avg_suff_stat = self.average_sufficient_statistic(sample)
        # add gaussian noise to mean parameters
        noise = jax.random.normal(key, shape=(self.dim,)) * shape + location
        noisy_means = avg_suff_stat + noise
        # convert to natural parameters
        return self.to_natural(noisy_means)

    # Templates

    def to_natural(self, means: Array) -> Array:
        """Convert mean to natural parameters via $\\theta = \\nabla\\phi(\\eta)$.

        Args:
            means: Mean parameters

        Returns:
            Natural parameters
        """
        return jax.value_and_grad(self.negative_entropy)(means)[1]

    def relative_entropy(self, p_means: Array, q_params: Array) -> Array:
        """Compute the entropy of $p$ relative to $q$ (a.k.a. KL divergence).

        $D(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx = \\theta \\cdot \\eta - \\psi(\\theta) - \\phi(\\eta)$, where

        - $p(x;\\eta)$ has mean parameters $\\eta$, and
        - $q(x;\\theta)$ has natural parameters $\\theta$.

        Args:
            p_means: Mean parameters of p
            q_params: Natural parameters of q

        Returns:
            KL divergence D(p || q) (scalar)
        """
        return (
            self.negative_entropy(p_means)
            + self.log_partition_function(q_params)
            - jnp.dot(q_params, p_means)
        )
