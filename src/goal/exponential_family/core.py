"""Core definitions for exponential families and their parameterizations.

This module defines the structure of an [`ExponentialFamily`][goal.exponential_family.core.ExponentialFamily] and their various parameter spaces, with a focus on the dually flat structure arising from convex conjugacy between the [`log_partition_function`][goal.exponential_family.core.Forward.log_partition_function] and the [`negative_entropy`][goal.exponential_family.Backward.negative_entropy].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold import Coordinates, Dual, Manifold, Point

### Coordinate Systems ###


# Coordinate systems for exponential families
class Mean(Coordinates):
    """Mean parameters $\\eta \\in \\text{H}$ given by expectations of sufficient statistics:

    $$\\eta = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]$$
    """

    ...


type Natural = Dual[Mean]
"""Natural parameters $\\theta \\in \\Theta$ defining an exponential family through:

$$p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))$$
"""

### Exponential Families ###


class ExponentialFamily(Manifold, ABC):
    """Base manifold class for exponential families.

    An exponential family is a manifold of probability distributions with densities
    of the form:

    $$
    p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
    $$

    where:

    * $\\theta \\in \\Theta$ are the natural parameters,
    * $\\mathbf s(x)$ is the sufficient statistic,
    * $\\mu(x)$ is the base measure, and
    * $\\psi(\\theta)$ is the log partition function.

    The base `ExponentialFamily` class includes methods that fundamentally define an exponential family, namely the base measure and sufficient statistic.
    """

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimension of the data space."""

    @abstractmethod
    def _compute_sufficient_statistic(self, x: Array) -> Array: ...

    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Convert observation to sufficient statistics.

        Maps a point $x$ to its sufficient statistic $\\mathbf s(x)$ in mean coordinates:

        $$\\mathbf s: \\mathcal{X} \\mapsto \\text{H}$$

        Args:
            x: Array containing a single observation

        Returns:
            Mean coordinates of the sufficient statistics
        """
        return Point(self._compute_sufficient_statistic(x))

    def average_sufficient_statistic(self, xs: Array) -> Point[Mean, Self]:
        """Average sufficient statistics of a batch of observations.

        Args:
            xs: Array of shape (batch_size, data_dim) containing : batch of observations

        Returns:
            Mean coordinates of average sufficient statistics
        """
        batch_stats = jax.vmap(self._compute_sufficient_statistic)(xs)
        avg_params = jnp.mean(batch_stats, axis=0)
        return Point(avg_params)

    @abstractmethod
    def log_base_measure(self, x: Array) -> Array:
        """Compute log of base measure $\\mu(x)$."""
        ...

    def natural_point(self, params: Array) -> Point[Natural, Self]:
        """Construct a point in natural coordinates."""
        return Point[Natural, Self](jnp.atleast_1d(params))

    def mean_point(self, params: Array) -> Point[Mean, Self]:
        """Construct a point in mean coordinates."""
        return Point[Mean, Self](jnp.atleast_1d(params))


class Generative(ExponentialFamily, ABC):
    """An `ExponentialFamily` that supports random sampling.

    Sampling is performed on distributions in natural coordinates. This enables estimation of mean parameters even when closed-form expressions for the log partition function are unavailable.
    """

    @abstractmethod
    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        """Generate random samples from the distribution.

        Args:
            p: Parameters in natural coordinates
            n: Number of samples to generate (default=1)

        Returns:
            Array of shape (n, *data_dims) containing samples
        """
        ...

    def stochastic_to_mean(
        self, key: Array, p: Point[Natural, Self], n: int
    ) -> Point[Mean, Self]:
        """Estimate mean parameters via Monte Carlo sampling."""
        samples = self.sample(key, p, n)
        return self.average_sufficient_statistic(samples)


class Forward(Generative, ABC):
    """Exponential family with an analytically tractable log-partition function, which thereby permits computing the expecting value of the sufficient statistic, and data-fitting via gradient descent.

    The log partition function $\\psi(\\theta)$ is given by:

    $$
    \\psi(\\theta) = \\log \\int_{\\mathcal{X}} \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x))dx
    $$

    Its gradient at a point in natural coordinates returns that point in mean coordinates:

    $$
    \\eta = \\nabla \\psi(\\theta) = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]
    $$
    """

    @abstractmethod
    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        """Internal method to compute $\\psi(\\theta)$."""
        ...

    def log_partition_function(self, p: Point[Natural, Self]) -> Array:
        """Compute log partition function $\\psi(\\theta)$."""
        return self._compute_log_partition_function(p.params)

    def to_mean(self, p: Point[Natural, Self]) -> Point[Mean, Self]:
        """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
        mean_params = jax.grad(self._compute_log_partition_function)(p.params)  # type: ignore
        return Point(mean_params)

    def log_density(self, p: Point[Natural, Self], x: Array) -> Array:
        """Compute log density at x.

        $$
        \\log p(x;\\theta) = \\theta \\cdot \\mathbf s(x) + \\log \\mu(x) - \\psi(\\theta)
        $$
        """
        suff_stats = self.sufficient_statistic(x)
        return (
            jnp.dot(p.params, suff_stats.params)
            + self.log_base_measure(x)
            - self.log_partition_function(p)
        )

    def density(self, p: Point[Natural, Self], x: Array) -> Array:
        """Compute density at x.

        $$
        p(x;\\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
        $$
        """
        return jnp.exp(self.log_density(p, x))


class Backward(Forward, ABC):
    """An exponential family comprising distributions for which the entropy can be evaluated in closed-form. The negative entropy is the convex conjugate of the log-partition function

    $$
    \\phi(\\eta) = \\sup_{\\theta} \\{\\theta \\cdot \\eta - \\psi(\\theta)\\},
    $$

    and its gradient at a point in mean coordinates returns that point in natural coordinates $\\theta = \\nabla\\phi(\\eta).$

     **NB:** This form of the negative entropy is the convex conjugate of the log-partition function, and does not factor in the base measure of the distribution. It may thus differ from the entropy as traditionally defined.
    """

    @abstractmethod
    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        """Internal method to compute $\\phi(\\eta)$."""
        ...

    def negative_entropy(self, p: Point[Mean, Self]) -> Array:
        """Compute negative entropy $\\phi(\\eta)$."""
        return self._compute_negative_entropy(p.params)

    def to_natural(self, p: Point[Mean, Self]) -> Point[Natural, Self]:
        """Convert mean to natural parameters via $\\theta = \\nabla\\phi(\\eta)$."""
        natural_params = jax.grad(self._compute_negative_entropy)(p.params)  # type: ignore
        return Point(natural_params)

    def relative_entropy(self, p: Point[Natural, Self], q: Point[Mean, Self]) -> Array:
        """Compute the entropy of $p$ relative to $q$.

        $D(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx = \\theta \\cdot \\eta - \\psi(\\theta) - \\phi(\\eta)$, where

        - $p(x;\\theta)$ has natural parameters $\\theta$,
        - $q(x;\\eta)$ has mean parameters $\\eta$.
        """
        return (
            self.log_partition_function(p)
            + self.negative_entropy(q)
            - jnp.dot(p.params, q.params)
        )
