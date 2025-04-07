"""This module implements the fundamental abstractions for exponential families, providing a hierarchy of classes with increasing capabilities for statistical modeling and inference.

See the package index for mathematical background on exponential families.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import (
    Coordinates,
    Dual,
    Manifold,
    Point,
    reduce_dual,
)
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
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Compute the sufficient statistics of an observation. In particular, maps a point $x$ to its sufficient statistic $\\mathbf s(x)$ in mean coordinates $\\mathbf s(x) \\in \\text{H}$"""

    @abstractmethod
    def log_base_measure(self, x: Array) -> Array:
        """Compute log of base measure $\\mu(x)$."""

    # Templates

    def average_sufficient_statistic(
        self, xs: Array, batch_size: int = 256
    ) -> Point[Mean, Self]:
        """Average sufficient statistics of a batch of observations."""

        def _sufficient_statistic(x: Array) -> Array:
            return self.sufficient_statistic(x).array

        return self.mean_point(batched_mean(_sufficient_statistic, xs, batch_size))

    def mean_point(self, params: Array) -> Point[Mean, Self]:
        """Construct a point in mean coordinates."""
        return self.point(params)

    def check_natural_parameters(self, params: Point[Natural, Self]) -> Array:
        """Check if parameters are valid for this exponential family."""
        return jnp.all(jnp.isfinite(params.array)).astype(jnp.int32)

    def natural_point(self, params: Array) -> Point[Natural, Self]:
        """Construct a point in natural coordinates."""
        return self.point(params)

    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Point[Natural, Self]:
        """Convenience function to randomly initialize the coordinates of a point based on a mean and a shape parameter --- by default this is a normal distribution, but may be overridden e.g. for bounded parameter spaces."""
        params = jax.random.normal(key, shape=(self.dim,)) * shape + location
        return self.natural_point(params)

    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Point[Natural, Self]:
        """Convenience function to initialize a model based on the sample. By default it ignores the sample, but most distributions can do better than this."""
        sample = sample
        return self.initialize(key, location, shape)


class Generative(ExponentialFamily, ABC):
    """Generative extends ExponentialFamily with sampling capabilities, enabling simulation-based methods like Monte Carlo estimation. This allows for working with distributions even when closed-form expressions for statistics are unavailable or intractable.

    In particular, sampling from $p(x; \\theta)$ provides an empirical approach to estimating expectation-based quantities like mean parameters:

    $$\\hat{\\eta} = \\frac{1}{n}\\sum_{i=1}^n \\mathbf{s}(x_i), \\quad x_i \\sim p(x; \\theta).$$
    """

    # Contract

    @abstractmethod
    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        """Generate random samples from the distribution."""

    # Templates

    def stochastic_to_mean(
        self, key: Array, params: Point[Natural, Self], n: int
    ) -> Point[Mean, Self]:
        """Estimate mean parameters via Monte Carlo sampling."""
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
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Compute log partition function $\\psi(\\theta)$."""

    # Templates

    def to_mean(self, params: Point[Natural, Self]) -> Point[Mean, Self]:
        """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
        return self.grad(self.log_partition_function, params)

    def log_density(self, params: Point[Natural, Self], x: Array) -> Array:
        """Compute log density at x.

        $$
        \\log p(x;\\theta) = \\theta \\cdot \\mathbf s(x) + \\log \\mu(x) - \\psi(\\theta)
        $$
        """
        suff_stats = self.sufficient_statistic(x)
        return (
            self.dot(params, suff_stats)
            + self.log_base_measure(x)
            - self.log_partition_function(params)
        )

    def density(self, params: Point[Natural, Self], x: Array) -> Array:
        """Compute density at x.

        $$
        p(x;\\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
        $$
        """
        return jnp.exp(self.log_density(params, x))

    def average_log_density(
        self, p: Point[Natural, Self], xs: Array, batch_size: int = 2048
    ) -> Array:
        """Compute average log density over a batch of observations.

        Returns:
            Scalar array of the average log density
        """

        def _log_density(x: Array) -> Array:
            return self.log_density(p, x)

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
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        """Compute negative entropy $\\phi(\\eta)$."""

    # Overrides

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Point[Natural, Self]:
        """Initialize a model based on the noisy average sufficient statistics."""
        avg_suff_stat = self.average_sufficient_statistic(sample)
        # add gaussian noise
        noise = jax.random.normal(key, shape=(self.dim,)) * shape + location
        avg_suff_stat = avg_suff_stat + self.point(noise)
        params = self.to_natural(avg_suff_stat)
        return self.natural_point(params.array)

    # Templates

    def to_natural(self, means: Point[Mean, Self]) -> Point[Natural, Self]:
        """Convert mean to natural parameters via $\\theta = \\nabla\\phi(\\eta)$."""
        return reduce_dual(self.grad(self.negative_entropy, means))

    def relative_entropy(
        self, p_means: Point[Mean, Self], q_params: Point[Natural, Self]
    ) -> Array:
        """Compute the entropy of $p$ relative to $q$ (a.k.a. KL divergence).

        $D(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx = \\theta \\cdot \\eta - \\psi(\\theta) - \\phi(\\eta)$, where

        - $p(x;\\eta)$ has mean parameters $\\eta$, and
        - $q(x;\\theta)$ has natural parameters $\\theta$.
        """
        return (
            self.negative_entropy(p_means)
            + self.log_partition_function(q_params)
            - self.dot(q_params, p_means)
        )
