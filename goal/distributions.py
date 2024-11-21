"""Core exponential families."""

from dataclasses import dataclass
from typing import Type

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from goal.exponential_family import ClosedFormExponentialFamily, Mean, Source
from goal.linear import Diagonal, PositiveDefinite, Scale
from goal.manifold import Coordinates, Point


@dataclass(frozen=True)
class MultivariateGaussian(ClosedFormExponentialFamily):
    """Multivariate Gaussian distributions.

    Parameters:
        data_dim: Dimension of the data space.
        covariance_shape: Covariance structure class (e.g. `Scale`, `Diagonal`, `PositiveDefinite`). Determines how the covariance matrix is parameterized.

    The standard expression for the Gaussian density in source coordinates is

    $$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}e^{-\\frac{1}{2}(x-\\mu) \\cdot \\Sigma^{-1} \\cdot (x-\\mu)}.$$

    where:

    - $\\mu$ is the mean vector,
    - $\\Sigma$ is the covariance matrix, and
    - $d$ is the dimension of the data.

    As an exponential family, the general form of the Gaussian family is defined by the base measure $\\mu(x) = -\\frac{d}{2}\\log(2\\pi)$ and sufficient statistic $\\mathbf{s}(x) = (x, x \\otimes x)$. The log-partition function is then given by

    $$\\psi(\\theta_1, \\theta_2) = -\\frac{1}{4}\\theta_1 \\cdot \\theta_2^{-1} \\cdot \\theta - \\frac{1}{2}\\log|-2\\theta_2|,$$

    and the negative entropy is given by

    $$\\psi(\\eta_1, \\eta_2) = -\\frac{1}{2}\\log|\\eta_2 - \\eta_1\\eta_1^T| - \\frac{d}{2}(1 + \\log(2\\pi)).$$

    Consequently, the mean parameters are given simply by the first and second moments $\\eta_1 = \\mu$ and $\\eta_2 = \\mu\\mu^T + \\Sigma$, and the natural parameters are given by $\\theta_1 = \\Sigma^{-1}\\mu$ and $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$.

    Finally, we handle different covariance structures (full, diagonal, scalar) through the `LinearOperator` hierarchy, so that the sufficient statistics and parameters are lowerer-dimensional, and the given manifold is a submanifold of the unrestricted family of Gaussians.


    """

    # Attributes

    data_dim: int
    """Dimension of the data space."""

    covariance_shape: Type[PositiveDefinite]
    """Covariance structure (e.g. `Scale`, `Diagonal`, `PositiveDefinite`)."""

    @property
    def covariance_dim(self) -> int:
        """Number of parameters needed for representing the covariance."""
        if issubclass(self.covariance_shape, Scale):
            return 1
        if issubclass(self.covariance_shape, Diagonal):
            return self.data_dim
        n = self.data_dim
        return (n * (n + 1)) // 2  # Triangular number for symmetric matrix

    @property
    def dimension(self) -> int:
        """Total dimension = `data_dim` + `covariance_dim`."""
        return self.data_dim + self.covariance_dim

    # Core class methods

    def _compute_sufficient_statistic(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        second_moment = self.covariance_shape.outer_product(x, x)
        return jnp.concatenate([x, second_moment.params])

    def log_base_measure(self, x: ArrayLike) -> Array:
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(self, natural_params: ArrayLike) -> Array:
        natural_params = jnp.asarray(natural_params)
        theta1, theta2 = self.split_params(Point(natural_params))
        return -0.25 * theta1 @ theta2.inverse() @ theta1 - 0.5 * (-2 * theta2).logdet()

    def _compute_negative_entropy(self, mean_params: ArrayLike) -> Array:
        mean_params = jnp.asarray(mean_params)
        mean, second_moment = self.split_params(Point(mean_params))

        # Compute covariance without forming full matrices
        outer_mean = self.covariance_shape.outer_product(mean, mean)
        covariance = second_moment - outer_mean

        return -0.5 * covariance.logdet() - 0.5 * self.data_dim * (
            1 + jnp.log(2 * jnp.pi)
        )

    # Additional methods

    def from_mean_and_covariance(
        self, mean: Array, covariance: PositiveDefinite
    ) -> Point[Source, "MultivariateGaussian"]:
        """Construct point from mean and covariance parameters in source coordinates."""
        params = jnp.concatenate([mean, covariance.params])
        return Point(params)

    def split_params(
        self, p: Point[Coordinates, "MultivariateGaussian"]
    ) -> tuple[Array, PositiveDefinite]:
        """Split parameters into location and scale components."""
        operator = self.covariance_shape.from_params(
            p.params[self.data_dim :], self.data_dim
        )
        return p.params[: self.data_dim], operator


@dataclass(frozen=True)
class Categorical(ClosedFormExponentialFamily):
    """Categorical distribution over $n$ states, of dimension $d = n-1$.

    Parameters:
        n_categories: Number of categories.

    For $n = (d+1)$ categories and label $0 \\leq k \\leq d$, the probability mass function is given by

    $$p(k; \\eta) = \\eta_k,$$

    where

    - $\\mathbf \\eta = (\\eta_1, \\ldots, \\eta_d)$ are the probabilities states 1 to $d$, and
    - $\\eta_0 = 1 - \\sum_{i=1}^d \\eta_i$ is the probability of state 0.

    The base measure is $\\mu(k) = 0$ for all $k$, and the sufficient statistic for $k=0$ is the 0 vector, and the one-hot vector for $k > 0$. The log-partition function is given by

    $$\\psi(\\theta) = \\log(1 + \\sum_{i=1}^d e^{\\theta_i}),$$

    and the negative entropy is given by

    $$\\phi(\\eta) = \\sum_{i=0}^d \\eta_i \\log(\\eta_i).$$

    As such, the the mapping from natural to mean parameters is given by $\\theta_i = \\log(\\eta_i/\\eta_0)$ for $i=1,\\ldots,d$. Finally, the source parameters are the same as the mean parameters.
    """

    n_categories: int

    @property
    def dimension(self) -> int:
        """Dimension $d$ is (`n_categories` - 1) due to the sum-to-one constraint."""
        return self.n_categories - 1

    def from_probs(self, probs: Array) -> Point[Mean, "Categorical"]:
        """Construct the mean parameters from the complete probabilities, dropping the first element."""
        return Point(probs[1:])

    def to_probs(self, p: Point[Mean, "Categorical"]) -> Array:
        """Return the probabilities of all labels."""
        probs = p.params
        prob0 = 1 - jnp.sum(probs)
        return jnp.concatenate([jnp.array([prob0]), probs])

    def _compute_sufficient_statistic(self, x: ArrayLike) -> Array:
        one_hot = jax.nn.one_hot(x, self.n_categories)
        return one_hot[1:]

    def log_base_measure(self, x: ArrayLike) -> Array:
        return jnp.array(0.0)

    def _compute_log_partition_function(self, natural_params: ArrayLike) -> Array:
        return jnp.log1p(jnp.sum(jnp.exp(natural_params)))

    def _compute_negative_entropy(self, mean_params: ArrayLike) -> Array:
        mean_params = jnp.asarray(mean_params)
        probs = self.to_probs(Point(mean_params))
        return jnp.sum(probs * jnp.log(probs))


@dataclass(frozen=True)
class Poisson(ClosedFormExponentialFamily):
    """
    The Poisson distribution over counts.

    The Poisson distribution is defined by a single rate parameter $\\eta > 0$. The probability mass function at count $k \\in \\mathbb{N}$ is given by

    $$p(k; \\eta) = \\frac{\\eta^k e^{-\\eta}}{k!}.$$

    The base measure is $\\mu(k) = -\\log(k!)$, and the sufficient statistic is simply the count itself. The log-partition function is given by $\\psi(\\theta) = e^{\\theta},$ and the negative entropy is given by $\\phi(\\eta) = \\eta\\log(\\eta) - \\eta$. As such, the mapping between the natural and mean parameters is given by $\\theta = \\log(\\eta)$. The source parameters ($\\lambda$) are the same as the mean parameters.
    """

    @property
    def dimension(self) -> int:
        """Single rate parameter."""
        return 1

    def _compute_sufficient_statistic(self, x: ArrayLike) -> Array:
        return jnp.asarray(x)

    def log_base_measure(self, x: ArrayLike) -> Array:
        k = jnp.asarray(x)
        return -jax.lax.lgamma(k + 1)

    def _compute_log_partition_function(self, natural_params: ArrayLike) -> Array:
        return jnp.exp(jnp.asarray(natural_params))

    def _compute_negative_entropy(self, mean_params: ArrayLike) -> Array:
        rate = jnp.asarray(mean_params)
        return rate * (jnp.log(rate) - 1)
