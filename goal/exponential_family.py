"""Core definitions for exponential families and their parameterizations.

This module defines the structure of exponential families and their various
parameter spaces, with a focus on the dually flat structure arising from
convex conjugacy between the log partition function and negative entropy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from goal.manifold import Coordinates, Manifold, Point


# Coordinate systems for exponential families
class Natural(Coordinates):
    """Natural parameters $\\theta \\in \\Theta$ defining an exponential family through:

    $$p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))$$
    """

    pass


class Mean(Coordinates):
    """Mean parameters $\\eta \\in \\text{H}$ given by expectations of sufficient statistics:

    $$\\eta = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]$$
    """

    pass


class Source(Coordinates):
    """Source parameters $\\omega \\in \\Omega$ representing conventional parameterizations."""

    pass


# Type variables for exponential family types
EF = TypeVar("EF", bound="ExponentialFamily")
DEF = TypeVar("DEF", bound="DifferentiableExponentialFamily")
CFEF = TypeVar("CFEF", bound="ClosedFormExponentialFamily")


@dataclass(frozen=True)
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

    The manifold structure includes:

    * Dimension of the parameter space
    * Maps between coordinate systems (Natural, Mean, Source)
    * Operations valid in each coordinate system
    """

    @abstractmethod
    def _compute_sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Internal method to compute sufficient statistics.

        Args:
            x: Array of shape (*data_dims) containing a single observation

        Returns:
            Array of sufficient statistics
        """
        pass

    def sufficient_statistic(self: EF, x: jnp.ndarray) -> Point[Mean, EF]:
        """Convert observation to sufficient statistics.

        Maps a point $x$ to its sufficient statistic $\\mathbf s(x)$ in mean coordinates:

        $$\\mathbf s: \\mathcal{X} \\mapsto \\text{H}$$

        Args:
            x: Array of shape (*data_dims) containing a single observation

        Returns:
            Mean coordinates of the sufficient statistics
        """
        return Point(self._compute_sufficient_statistic(x), self)

    def average_sufficient_statistic(self: EF, xs: jnp.ndarray) -> Point[Mean, EF]:
        """Average sufficient statistics of a batch of observations.

        Args:
            xs: Array of shape (batch_size, *data_dims) containing observations

        Returns:
            Mean coordinates of average sufficient statistics
        """
        batch_stats = jax.vmap(self.sufficient_statistic)(xs)
        avg_params = jnp.mean(batch_stats.params, axis=0)
        return Point(avg_params, self)

    @abstractmethod
    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log of base measure $\\mu(x)$."""
        pass


@dataclass(frozen=True)
class DifferentiableExponentialFamily(ExponentialFamily, ABC):
    """Exponential family with differentiable log partition function.

    The log partition function $\\psi(\\theta)$ is given by:

    $$
    \\psi(\\theta) = \\log \\int_{\\mathcal{X}} \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x))dx
    $$

    Its gradient gives the mean parameters:

    $$
    \\eta = \\nabla \\psi(\\theta) = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]
    $$
    """

    @abstractmethod
    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Internal method to compute $\\psi(\\theta)$."""
        pass

    def log_partition_function(self: DEF, p: Point[Natural, DEF]) -> jnp.ndarray:
        """Compute log partition function $\\psi(\\theta)$."""
        return self._compute_log_partition_function(p.params)

    def to_mean(self: DEF, p: Point[Natural, DEF]) -> Point[Mean, DEF]:
        """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
        mean_params = jax.grad(self._compute_log_partition_function)(p.params)
        return Point(mean_params, self)

    def log_density(self: DEF, p: Point[Natural, DEF], x: jnp.ndarray) -> jnp.ndarray:
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


@dataclass(frozen=True)
class ClosedFormExponentialFamily(DifferentiableExponentialFamily, ABC):
    """Exponential family with closed form entropy and parameter conversions.

    For many exponential families, the entropy has a closed form in terms
    of mean parameters. The negative entropy is the convex conjugate of the
    log partition function:

    $$
    -\\text{H}(\\eta) = \\sup_{\\theta} \\{\\theta \\cdot \\eta - \\psi(\\theta)\\}
    $$

    The gradient relationship between natural and mean parameters is symmetric:

    $$
    \\theta = \\nabla(-\\text{H}(\\eta))
    $$

    $$
    \\eta = \\nabla \\psi(\\theta)
    $$
    """

    @abstractmethod
    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Internal method to compute $-\\text{H}(\\eta)$."""
        pass

    def negative_entropy(self: CFEF, p: Point[Mean, CFEF]) -> jnp.ndarray:
        """Compute negative entropy $-\\text{H}(\\eta)$."""
        return self._compute_negative_entropy(p.params)

    def to_natural(self: CFEF, p: Point[Mean, CFEF]) -> Point[Natural, CFEF]:
        """Convert mean to natural parameters via $\\theta = \\nabla(-\\text{H}(\\eta))$."""
        natural_params = jax.grad(self._compute_negative_entropy)(p.params)
        return Point(natural_params, self)


class Generative(ABC):
    """Mixin for distributions that support random sampling.

    Designed to work with JAX's random number generation system.
    """

    @abstractmethod
    def sample(self, p: Point, key: PRNGKeyArray, n: int = 1) -> jnp.ndarray:
        """Generate random samples from the distribution.

        Args:
            p: Parameters of the distribution
            key: JAX random key
            n: Number of samples to generate (default=1)

        Returns:
            Array of shape (n, *event_shape) containing samples
        """
        pass
