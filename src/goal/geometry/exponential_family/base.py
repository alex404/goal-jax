"""This module implements the fundamental abstractions for exponential families, providing a hierarchy of classes with increasing capabilities for statistical modeling and inference.

See the package index for mathematical background on exponential families.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Coordinates, Dual, Manifold
from ..manifold.util import batched_mean

### Coordinate Systems ###


# Coordinate systems for exponential families
class Natural(Coordinates):
    """Natural serves as a type tag for parameters that directly parameterize the exponential family density. These parameters are used in maximum likelihood estimation and directly control the shape of the distribution.

    In theory, natural parameters $\\theta \\in \\Theta$ define an exponential family through:

    $$p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))$$

    where $\\Theta$ is a convex subset of $\\mathbb{R}^d$.
    """


type Mean = Dual[Natural]

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

    def check_natural_parameters(self, natural_params: Array) -> Array:
        """Check if parameters are valid for this exponential family.

        Args:
            natural_params: Natural parameters

        Returns:
            Scalar indicating validity (all finite)
        """
        return jnp.all(jnp.isfinite(natural_params)).astype(jnp.int32)

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
    def sample(self, key: Array, natural_params: Array, n: int = 1) -> Array:
        """Generate random samples from the distribution.

        Args:
            key: JAX random key
            natural_params: Natural parameters
            n: Number of samples

        Returns:
            Array of n samples
        """

    # Templates

    def stochastic_to_mean(self, key: Array, natural_params: Array, n: int) -> Array:
        """Estimate mean parameters via Monte Carlo sampling.

        Args:
            key: JAX random key
            natural_params: Natural parameters
            n: Number of samples

        Returns:
            Estimated mean parameters
        """
        samples = self.sample(key, natural_params, n)
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
    def log_partition_function(self, natural_params: Array) -> Array:
        """Compute log partition function $\\psi(\\theta)$.

        Args:
            natural_params: Natural parameters

        Returns:
            Log partition function value (scalar)
        """

    # Templates

    def to_mean(self, natural_params: Array) -> Array:
        """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$.

        Args:
            natural_params: Natural parameters

        Returns:
            Mean parameters
        """
        return self.grad(self.log_partition_function, natural_params)

    def log_density(self, natural_params: Array, x: Array) -> Array:
        """Compute log density at x.

        $$
        \\log p(x;\\theta) = \\theta \\cdot \\mathbf s(x) + \\log \\mu(x) - \\psi(\\theta)
        $$

        Args:
            natural_params: Natural parameters
            x: Data point

        Returns:
            Log density (scalar)
        """
        suff_stats = self.sufficient_statistic(x)
        return (
            self.dot(natural_params, suff_stats)
            + self.log_base_measure(x)
            - self.log_partition_function(natural_params)
        )

    def density(self, natural_params: Array, x: Array) -> Array:
        """Compute density at x.

        $$
        p(x;\\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
        $$

        Args:
            natural_params: Natural parameters
            x: Data point

        Returns:
            Density (scalar)
        """
        return jnp.exp(self.log_density(natural_params, x))

    def average_log_density(
        self, natural_params: Array, xs: Array, batch_size: int = 2048
    ) -> Array:
        """Compute average log density over a batch of observations.

        Args:
            natural_params: Natural parameters
            xs: Batch of data points
            batch_size: Size of batches for vmapped computation

        Returns:
            Scalar array of the average log density
        """

        def _log_density(x: Array) -> Array:
            return self.log_density(natural_params, x)

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
    def negative_entropy(self, mean_params: Array) -> Array:
        """Compute negative entropy $\\phi(\\eta)$.

        Args:
            mean_params: Mean parameters

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
        noisy_mean_params = avg_suff_stat + noise
        # convert to natural parameters
        return self.to_natural(noisy_mean_params)

    # Templates

    def to_natural(self, mean_params: Array) -> Array:
        """Convert mean to natural parameters via $\\theta = \\nabla\\phi(\\eta)$.

        Args:
            mean_params: Mean parameters

        Returns:
            Natural parameters
        """
        return self.grad(self.negative_entropy, mean_params)

    def relative_entropy(self, p_mean_params: Array, q_natural_params: Array) -> Array:
        """Compute the entropy of $p$ relative to $q$ (a.k.a. KL divergence).

        $D(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx = \\theta \\cdot \\eta - \\psi(\\theta) - \\phi(\\eta)$, where

        - $p(x;\\eta)$ has mean parameters $\\eta$, and
        - $q(x;\\theta)$ has natural parameters $\\theta$.

        Args:
            p_mean_params: Mean parameters of p
            q_natural_params: Natural parameters of q

        Returns:
            KL divergence D(p || q) (scalar)
        """
        return (
            self.negative_entropy(p_mean_params)
            + self.log_partition_function(q_natural_params)
            - self.dot(q_natural_params, p_mean_params)
        )
