import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto

import jax
import jax.numpy as jnp


class ParameterType(Enum):
    NATURAL = auto()
    MEAN = auto()
    SOURCE = auto()


@dataclass(frozen=True)
class ExponentialFamily(ABC):
    """Base class for exponential families with flat parameter arrays"""

    params: jnp.ndarray
    _param_type: ParameterType

    @abstractmethod
    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert point to sufficient statistics (flat array)"""
        pass

    @abstractmethod
    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Log base measure"""
        pass

    def dimension(self) -> int:
        return len(self.params)


@dataclass(frozen=True)
class DifferentiableExponentialFamily(ExponentialFamily, ABC):
    """Exponential family that we can optimize with gradient descent"""

    @abstractmethod
    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log partition function for given natural parameters"""
        pass

    def log_partition_function(self) -> jnp.ndarray:
        """Log partition function evaluated at current parameters"""
        if self._param_type != ParameterType.NATURAL:
            raise ValueError(
                "Log partition function must be called in natural coordinates"
            )
        return self._compute_log_partition_function(self.params)

    def to_mean(self) -> ExponentialFamily:
        """Convert to mean parameters by differentiating log partition function"""
        if self._param_type == ParameterType.MEAN:
            logging.warning("Already in mean coordinates, no conversion needed")
            return self

        mean_params = jax.grad(self._compute_log_partition_function)(self.params)
        return replace(self, params=mean_params, _param_type=ParameterType.MEAN)

    def log_density(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log density at point x"""
        if self._param_type != ParameterType.NATURAL:
            raise ValueError(
                "Log density function must be called in natural coordinates"
            )
        return (
            jnp.dot(self.params, self.sufficient_statistic(x))
            + self.log_base_measure(x)
            - self.log_partition_function()
        )


@dataclass(frozen=True)
class ClosedFormExponentialFamily(DifferentiableExponentialFamily, ABC):
    """Exponential family with closed form entropy"""

    @abstractmethod
    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute negative entropy for given mean parameters"""
        pass

    def negative_entropy(self) -> jnp.ndarray:
        """Compute negative entropy at current parameters"""
        if self._param_type != ParameterType.MEAN:
            raise ValueError("Negative entropy must be called in mean coordinates")
        return self._compute_negative_entropy(self.params)

    def to_natural(self) -> ExponentialFamily:
        """Convert to mean parameters by differentiating log partition function"""
        if self._param_type == ParameterType.NATURAL:
            logging.warning("Already in natural coordinates, no conversion needed")
            return self

        natural_params = jax.grad(self._compute_negative_entropy)(self.params)
        return replace(self, params=natural_params, _param_type=ParameterType.NATURAL)


@dataclass(frozen=True)
class MultivariateGaussian(ClosedFormExponentialFamily):
    """Multivariate Gaussian distribution"""

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns flat array of [x, vec(xx^T)]"""
        return jnp.concatenate([x, jnp.outer(x, x).ravel()])

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log partition function for given natural parameters"""
        eta = natural_params[: self.dim]
        theta = natural_params[self.dim :].reshape(self.dim, self.dim)
        precision = -2 * theta
        return (
            0.25 * jnp.dot(eta, jnp.linalg.solve(precision, eta))
            - 0.5 * jnp.linalg.slogdet(precision)[1]
        )

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute negative entropy for given mean parameters"""
        mean = mean_params[: self.dim]
        second_moment = mean_params[self.dim :].reshape(self.dim, self.dim)
        covariance = second_moment - jnp.outer(mean, mean)
        return -0.5 * jnp.linalg.slogdet(covariance)[1] - 0.5 * self.dim * (
            1 + jnp.log(2 * jnp.pi)
        )


@dataclass(frozen=True)
class Categorical(ClosedFormExponentialFamily):
    """Categorical distribution as exponential family"""

    n_categories: int

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """One-hot encoding"""
        return jax.nn.one_hot(x[0], self.n_categories)  # Take first element for scalar

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Base measure is constant 1, so log is 0"""
        return jnp.array(0.0)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Log sum exp of natural parameters"""
        return jax.nn.logsumexp(natural_params)

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute negative entropy for given mean parameters"""
        return jnp.sum(mean_params * jnp.log(mean_params))
