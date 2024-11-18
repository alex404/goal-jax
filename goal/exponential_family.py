"""Core definitions for exponential families and their parameterizations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TypeVar

import jax
import jax.numpy as jnp

from goal.manifold import Manifold

### Core Definitions ###


class Parameterization(Enum):
    """Parameter space representation."""

    NATURAL = auto()
    MEAN = auto()
    SOURCE = auto()


EF = TypeVar("EF", bound="ExponentialFamily")


@dataclass(frozen=True)
class ExponentialFamily(Manifold, ABC):
    """Base class for exponential families.

    An exponential family is a family of probability distributions whose density
    has the form:

    $$
    p(x; \\theta) = h(x)\\exp(\\langle\\theta, T(x)\\rangle - A(\\theta))
    $$

    where:
    * $\\theta \\in \\Theta$ are the natural parameters
    * $T(x)$ is the sufficient statistic
    * $h(x)$ is the base measure
    * $A(\\theta)$ is the log partition function
    """

    parameterization: Parameterization

    @abstractmethod
    def sufficient_statistic(self: EF, x: jnp.ndarray) -> "EF":
        """Convert point to sufficient statistics (flat array).

        Maps a point x to its sufficient statistic T(x), where:

        $$
        T: \\mathcal{X} \\to \\mathbb{R}^d
        $$
        """
        ...

    @abstractmethod
    def average_sufficient_statistic(self: EF, xs: jnp.ndarray) -> "EF":
        """Compute mean sufficient statistics (MLE) from data.

        Args:
            xs: Array of shape (batch_size, *data_dims) containing observations

        Returns:
            An ExponentialFamily object in mean coordinates corresponding to
            the average sufficient statistics of xs
        """
        batch_sufficient_stats = jax.vmap(self.sufficient_statistic)(xs)
        avg_params = jnp.mean(batch_sufficient_stats.params, axis=0)
        return replace(self, params=avg_params, parameterization=Parameterization.MEAN)

    @abstractmethod
    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log of base measure h(x)."""
        ...


DEF = TypeVar("DEF", bound="DifferentiableExponentialFamily")


@dataclass(frozen=True)
class DifferentiableExponentialFamily(ExponentialFamily, ABC):
    """Exponential family with differentiable log partition function.

    The log partition function A(θ) is given by:

    $$
    A(\\theta) = \\log \\int_{\\mathcal{X}} h(x)\\exp(\\langle\\theta, T(x)\\rangle)dx
    $$

    The gradient of A(θ) gives the mean parameters:

    $$
    \\eta = \\nabla A(\\theta) = \\mathbb{E}_{p(x;\\theta)}[T(x)]
    $$
    """

    @abstractmethod
    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log partition function A(θ) for given natural parameters."""
        pass

    def log_partition_function(self) -> jnp.ndarray:
        """Log partition function A(θ) evaluated at current parameters.

        Note:
            Must be called in natural parameterization.
        """
        if self.parameterization != Parameterization.NATURAL:
            raise ValueError("Log partition function requires natural parameters")
        return self._compute_log_partition_function(self.params)

    def to_mean(self: DEF) -> "DEF":
        """Convert to mean parameterization η = ∇A(θ).

        The mean parameters are obtained by differentiating the log partition function:

        $$
        \\eta = \\nabla A(\\theta)
        $$

        This relationship follows from the fact that A(θ) is the cumulant generating
        function of the sufficient statistics.
        """
        if self.parameterization == Parameterization.MEAN:
            logging.warning("Already in mean coordinates")
            return self
        mean_params = jax.grad(self._compute_log_partition_function)(self.params)
        return replace(self, params=mean_params, _param_type=Parameterization.MEAN)

    def log_density(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log density at point x.

        $$
        \\log p(x;\\theta) = \\langle\\theta, T(x)\\rangle + \\log h(x) - A(\\theta)
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


CFEF = TypeVar("CFEF", bound="ClosedFormExponentialFamily")


@dataclass(frozen=True)
class ClosedFormExponentialFamily(DifferentiableExponentialFamily, ABC):
    """Exponential family with closed form entropy and parameter conversions.

    For many common exponential families, the entropy has a closed form in terms
    of the mean parameters. The negative entropy is the convex conjugate of the
    log partition function:

    $$
    -H(\\eta) = \\sup_{\\theta} \\{\\langle\\theta, \\eta\\rangle - A(\\theta)\\}
    $$

    The gradient relationship between natural and mean parameters is symmetric:

    $$
    \\theta = \\nabla(-H(\\eta))
    $$

    $$
    \\eta = \\nabla A(\\theta)
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
        """Convert to natural parameterization θ = ∇(-H(η)).

        The natural parameters are obtained by differentiating the negative entropy:

        $$
        \\theta = \\nabla(-H(\\eta))
        $$

        This relationship follows from the convex duality between the log partition
        function and the negative entropy.
        """
        if self.parameterization == Parameterization.NATURAL:
            logging.warning("Already in natural coordinates")
            return self
        natural_params = jax.grad(self._compute_negative_entropy)(self.params)
        return replace(
            self, params=natural_params, _param_type=Parameterization.NATURAL
        )
