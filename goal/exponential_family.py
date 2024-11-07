from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp


class ExponentialFamily(eqx.Module):
    """Base class for exponential families with flat parameter arrays"""

    @abstractmethod
    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert point to sufficient statistics (flat array)"""
        pass

    @abstractmethod
    def log_base_measure(self, x: jnp.ndarray) -> float:
        """Log base measure"""
        pass

    @abstractmethod
    def potential(self, natural_params: jnp.ndarray) -> float:
        """Log partition function"""
        pass

    @abstractmethod
    def to_natural(self) -> jnp.ndarray:
        """Convert to natural parameters"""
        pass

    def log_density(self, x: jnp.ndarray) -> float:
        """Compute log density at point x"""
        params = self.to_natural()
        return float(
            jnp.dot(params, self.sufficient_statistic(x))
            + self.log_base_measure(x)
            - self.potential(params)
        )


class MultivariateGaussian(ExponentialFamily):
    """Multivariate Gaussian distribution"""

    dim: int
    mean: jnp.ndarray
    covariance: jnp.ndarray

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns flat array of [x, vec(xx^T)]"""
        return jnp.concatenate([x, jnp.outer(x, x).ravel()])

    def log_base_measure(self, x: jnp.ndarray) -> float:
        return float(-0.5 * self.dim * jnp.log(2 * jnp.pi))

    def to_natural(self) -> jnp.ndarray:
        """Convert to natural parameters [η, vec(Θ)]"""
        precision = jnp.linalg.inv(self.covariance)
        eta = jnp.dot(precision, self.mean)
        theta = -0.5 * precision
        return jnp.concatenate([eta, theta.ravel()])

    def potential(self, natural_params: jnp.ndarray) -> float:
        """Compute log partition function"""
        eta = natural_params[: self.dim]
        theta = natural_params[self.dim :].reshape(self.dim, self.dim)
        precision = -2 * theta
        return float(
            0.25 * jnp.dot(eta, jnp.linalg.solve(precision, eta))
            - 0.5 * jnp.linalg.slogdet(precision)[1]
        )

    def log_density(self, x: jnp.ndarray) -> float:
        """Optimized log density computation"""
        diff = x - self.mean
        return float(
            -0.5 * jnp.dot(diff, jnp.linalg.solve(self.covariance, diff))
            - 0.5 * jnp.linalg.slogdet(self.covariance)[1]
            - 0.5 * self.dim * jnp.log(2 * jnp.pi)
        )
