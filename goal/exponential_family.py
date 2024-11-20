"""Core definitions for exponential families and their parameterizations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Type, TypeVar

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from goal.manifold import Manifold

### TypeVars ###


EF = TypeVar("EF", bound="ExponentialFamily")
DEF = TypeVar("DEF", bound="DifferentiableExponentialFamily")
CFEF = TypeVar("CFEF", bound="ClosedFormExponentialFamily")

### Core Definitions ###


class Parameterization(Enum):
    """Standard parameter spaces for exponential families:

    * NATURAL: Natural parameters $\\theta \\in \\Theta$
    * MEAN: Mean parameters $\\eta \\in \\text{H}$
    * SOURCE: Source parameters $\\omega \\in \\Omega$

    The source parameterization to represent the conventional parameters of a distribution e.g. mean and covariance of a Gaussian.

    """

    NATURAL = auto()
    MEAN = auto()
    SOURCE = auto()


@dataclass(frozen=True)
class ExponentialFamily(Manifold, ABC):
    """Base class for exponential families.

    An exponential family is a family of probability distributions whose density
    has the form:

    $$
    p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
    $$

    where:

    * $\\theta \\in \\Theta$ are the natural parameters
    * $\\mathbf s(x)$ is the sufficient statistic
    * $\\mu(x)$ is the base measure
    * $\\psi(\\theta)$ is the log partition function
    """

    parameterization: Parameterization

    @classmethod
    @abstractmethod
    def _compute_sufficient_statistic(
        cls: Type[EF], x: jnp.ndarray, **hyperparams
    ) -> jnp.ndarray:
        """Convert point to sufficient statistics (flat array). This is the internal implementation."""
        ...

    @classmethod
    def sufficient_statistic(cls: Type[EF], x: jnp.ndarray, **hyperparams) -> EF:
        """Convert point to sufficient statistics.

        Maps a point $x$ to its sufficient statistic $\\mathbf s(x)$ in the mean parameter space, where:

        $$
        \\mathbf s: \\mathcal{X} \\mapsto \\text{H}
        $$

        Args:
            x: Array of shape (*data_dims) containing a single observation
            **hyperparams: Additional parameters for the distribution
        Returns:
            Mean coordinates of the sufficient statistics of x
        """
        return cls(
            params=cls._compute_sufficient_statistic(x, **hyperparams),
            parameterization=Parameterization.MEAN,
            **hyperparams,
        )

    @classmethod
    def average_sufficient_statistic(
        cls: Type[EF], xs: jnp.ndarray, **hyperparams
    ) -> "EF":
        """Take average of the sufficient statistics of a batch of observations.

        Args:
            xs: Array of shape (batch_size, *data_dims) containing observations
            **hyperparams: Additional parameters for the distribution

        Returns:
            Mean coordinates corresponding of the average sufficient statistics of xs
        """
        batch_sufficient_stats = jax.vmap(cls.sufficient_statistic)(xs)
        avg_params = jnp.mean(batch_sufficient_stats.params, axis=0)
        return cls(
            params=avg_params, parameterization=Parameterization.MEAN, **hyperparams
        )

    @abstractmethod
    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log of base measure $\\mu(x)$."""
        ...


@dataclass(frozen=True)
class DifferentiableExponentialFamily(ExponentialFamily, ABC):
    """Exponential family with differentiable log partition function.

    The log partition function $\\psi(\\theta)$ is given by:

    $$
    \\psi(\\theta) = \\log \\int_{\\mathcal{X}} \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x))dx
    $$

    The gradient of $\\psi(\\theta)$ gives the mean parameters:

    $$
    \\eta = \\nabla \\psi(\\theta) = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]
    $$
    """

    @abstractmethod
    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log partition function $\\psi(\\theta)$ for given natural parameters."""
        pass

    def log_partition_function(self) -> jnp.ndarray:
        """Log partition function $\\psi(\\theta)$ evaluated at current parameters.

        Note:
            Must be called in natural parameterization.
        """
        if self.parameterization != Parameterization.NATURAL:
            raise ValueError("Log partition function requires natural parameters")
        return self._compute_log_partition_function(self.params)

    def to_mean(self: DEF) -> "DEF":
        """Convert to mean parameterization $\\eta = \\nabla \\psi(\\theta)$.

        The mean parameters are obtained by differentiating the log partition function:

        $$
        \\eta = \\nabla \\psi(\\theta)
        $$

        This relationship follows from the fact that $\\psi(\\theta)$ is the cumulant generating
        function of the sufficient statistics.
        """
        if self.parameterization == Parameterization.MEAN:
            logging.warning("Already in mean coordinates")
            return self
        mean_params = jax.grad(self._compute_log_partition_function)(self.params)
        return replace(self, params=mean_params, parameterization=Parameterization.MEAN)

    def log_density(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log density at point $x$.

        $$
        \\log p(x;\\theta) = \\theta \\cdot \\mathbf s(x) + \\log \\mu(x) - \\psi(\\theta)
        $$

        Note:
            Must be called in natural parameterization.
        """
        if self.parameterization != Parameterization.NATURAL:
            raise ValueError("Log density requires natural parameters")
        return (
            jnp.dot(self.params, self.sufficient_statistic(x).params)
            + self.log_base_measure(x)
            - self.log_partition_function()
        )


@dataclass(frozen=True)
class ClosedFormExponentialFamily(DifferentiableExponentialFamily, ABC):
    """Exponential family with closed form entropy and parameter conversions.

    For many common exponential families, the entropy has a closed form in terms
    of the mean parameters. The negative entropy is the convex conjugate of the
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

    This duality allows efficient conversion between parameterizations.
    """

    @abstractmethod
    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute negative entropy for given mean parameters."""
        pass

    def negative_entropy(self) -> jnp.ndarray:
        """Compute negative entropy at current parameters.

        Note:
            Must be called in mean parameterization.
        """
        if self.parameterization != Parameterization.MEAN:
            raise ValueError("Negative entropy requires mean parameters")
        return self._compute_negative_entropy(self.params)

    def to_natural(self: CFEF) -> "CFEF":
        """Convert to natural parameterization $\\theta = \\nabla(-\\text{H}(\\eta))$.

        The natural parameters are obtained by differentiating the negative entropy:

        $$
        \\theta = \\nabla(-\\text{H}(\\eta))
        $$

        This relationship follows from the convex duality between the log partition
        function and the negative entropy.
        """
        if self.parameterization == Parameterization.NATURAL:
            logging.warning("Already in natural coordinates")
            return self
        natural_params = jax.grad(self._compute_negative_entropy)(self.params)
        return replace(
            self, params=natural_params, parameterization=Parameterization.NATURAL
        )


class Generative(ABC):
    """Adds sampling capabilities to probability distributions.

    This mixin class provides an interface for generating random samples from
    probability distributions. It is designed to work with JAX's random number
    generation system.
    """

    @abstractmethod
    def sample(self, key: PRNGKeyArray, n: int = 1) -> jnp.ndarray:
        """Generate random samples from the distribution.

        Args:
            key: JAX random key for random number generation
            n: Number of samples to generate (default=1)

        Returns:
            Array of samples. If n=1, shape is event_shape.
            Otherwise shape is (n, *event_shape).
        """
        pass
