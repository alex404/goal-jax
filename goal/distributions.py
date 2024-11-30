"""Core exponential families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from goal.exponential_family import ClosedForm, Generative, Mean, Natural
from goal.linear import (
    Diagonal,
    Identity,
    PositiveDefinite,
    PositiveDefiniteMap,
    Scale,
    TypeMap,
)
from goal.manifold import Coordinates, Point

C = TypeVar("C", bound=Coordinates)
PD = TypeVar("PD", bound=PositiveDefinite)

type FullNormal = Normal[PositiveDefinite]
type DiagonalNormal = Normal[Diagonal]
type IsotropicNormal = Normal[Scale]
type StandardNormal = Normal[Identity]

N = TypeVar("N", bound=Normal[PositiveDefinite])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Normal[R: PositiveDefinite](ClosedForm, Generative):
    """(Multivariate) Normal distributions.

    Parameters:
        data_dim: Dimension of the data space.
        covariance_shape: Covariance structure class (e.g. `Scale`, `Diagonal`, `PositiveDefinite`). Determines how the covariance matrix is parameterized.

    The standard expression for the Normal density is

    $$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}e^{-\\frac{1}{2}(x-\\mu) \\cdot \\Sigma^{-1} \\cdot (x-\\mu)}.$$

    where

    - $\\mu$ is the mean vector,
    - $\\Sigma$ is the covariance matrix, and
    - $d$ is the dimension of the data.

    As an exponential family, the general form of the Normal family is defined by the base measure $\\mu(x) = -\\frac{d}{2}\\log(2\\pi)$ and sufficient statistic $\\mathbf{s}(x) = (x, x \\otimes x)$. The log-partition function is then given by

    $$\\psi(\\theta_1, \\theta_2) = -\\frac{1}{4}\\theta_1 \\cdot \\theta_2^{-1} \\cdot \\theta - \\frac{1}{2}\\log|-2\\theta_2|,$$

    and the negative entropy is given by

    $$\\psi(\\eta_1, \\eta_2) = -\\frac{1}{2}\\log|\\eta_2 - \\eta_1\\eta_1^T| - \\frac{d}{2}(1 + \\log(2\\pi)).$$

    Consequently, the mean parameters are given simply by the first and second moments $\\eta_1 = \\mu$ and $\\eta_2 = \\mu\\mu^T + \\Sigma$, and the natural parameters are given by $\\theta_1 = \\Sigma^{-1}\\mu$ and $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$.

    Finally, we handle different covariance structures (full, diagonal, scalar) through the `LinearOperator` hierarchy, so that the sufficient statistics and parameters are lowerer-dimensional, and the given manifold is a submanifold of the unrestricted family of Normals.
    """

    # Attributes

    covariance_shape: PositiveDefiniteMap[R, StandardNormal]
    """Covariance structure (e.g. `Scale`, `Diagonal`, `PositiveDefinite`)."""

    @property
    def data_dim(self) -> int:
        """Dimension of the data space."""
        return self.covariance_shape.shape[0]

    @property
    def dimension(self) -> int:
        """Total dimension = `data_dim` + `covariance_dim`."""
        return self.data_dim + self.covariance_shape.dimension

    # Core class methods

    def _compute_sufficient_statistic(self: Normal[R], x: ArrayLike) -> Array:
        x = jnp.atleast_1d(x)

        # Create point in StandardNormal
        x_point: Point[Mean, StandardNormal] = Point(x)

        # Compute outer product with appropriate structure
        second_moment: Point[
            TypeMap[Mean, Natural], PositiveDefiniteMap[R, StandardNormal]
        ] = self.covariance_shape.outer_product(x_point, x_point)

        # Concatenate components into parameter vector
        return jnp.concatenate([x_point.params, second_moment.params])

    # def log_base_measure(self, x: ArrayLike) -> Array:
    #     return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)
    #
    # def _compute_log_partition_function(self, natural_params: Array) -> Array:
    #     natural_params = jnp.asarray(natural_params)
    #     theta1, theta2 = self.split_natural_params(Point(natural_params))
    #
    #     precision = -2 * theta2
    #     covariance = precision.inverse()  # This is Σ
    #     mean = covariance.matvec(theta1)
    #
    #     return 0.5 * jnp.dot(theta1, mean) - 0.5 * precision.logdet()
    #
    # def _compute_negative_entropy(self, mean_params: ArrayLike) -> Array:
    #     mean_params = jnp.asarray(mean_params)
    #     mean, second_moment = self.split_mean_params(Point(mean_params))
    #
    #     # Compute covariance without forming full matrices
    #     outer_mean = self.covariance_shape.outer_product(mean, mean)
    #     covariance = second_moment - outer_mean
    #
    #     log_det = covariance.logdet()
    #     entropy = 0.5 * (self.data_dim + self.data_dim * jnp.log(2 * jnp.pi) + log_det)
    #
    #     return -entropy
    #
    # def sample(
    #     self: Normal,
    #     key: ArrayLike,
    #     p: Point[Natural, Normal],
    #     n: int = 1,
    # ) -> Array:
    #     mean_point = self.to_mean(p)
    #     mean, covariance = self.to_mean_and_covariance(mean_point)
    #
    #     # Draw standard normal samples and transform
    #     key = jnp.asarray(key)
    #     shape = (n, self.data_dim)
    #     z = jax.random.normal(key, shape)
    #
    #     # Transform using Cholesky for numerical stability
    #     chol = covariance.cholesky
    #     return mean + (chol @ z.T).T
    #
    # # Additional methods
    #
    # def _split_params(self, p: Point[C, Normal]) -> tuple[Array, PositiveDefinite]:
    #     """Split parameters into location and scale components."""
    #     params = p.params[self.data_dim :]
    #
    #     operator = self.covariance_shape(params, self.data_dim)
    #     return p.params[: self.data_dim], operator
    #
    # def _join_params(self, mean: Array, covariance: PositiveDefinite) -> Array:
    #     """Join location and scale parameters into a single array."""
    #     return jnp.concatenate([mean, covariance.params])
    #
    # def from_mean_and_covariance(
    #     self, mean: Array, covariance: PositiveDefinite
    # ) -> Point[Mean, Normal]:
    #     """Construct a `Point` in `Mean` coordinates from the mean $\\mu$ and covariance $\\Sigma$."""
    #     second_moment = self.covariance_shape.outer_product(mean, mean) + covariance
    #     params = self._join_params(mean, second_moment)
    #     return Point(params)
    #
    # def to_mean_and_covariance(
    #     self, p: Point[Mean, Normal]
    # ) -> tuple[Array, PositiveDefinite]:
    #     """Extract the mean $\\mu$ and covariance $\\Sigma$ from a `Point` in `Mean` coordinates."""
    #     mean, second_moment = self.split_mean_params(p)
    #     covariance = second_moment - self.covariance_shape.outer_product(mean, mean)
    #     return mean, covariance
    #
    # def split_mean_params(
    #     self, p: Point[Mean, Normal]
    # ) -> tuple[Array, PositiveDefinite]:
    #     return self._split_params(p)
    #
    # def join_mean_params(
    #     self, mean: Array, covariance: PositiveDefinite
    # ) -> Point[Mean, Normal]:
    #     return Point(self._join_params(mean, covariance))
    #
    # def split_natural_params(
    #     self, p: Point[Natural, Normal]
    # ) -> tuple[Array, PositiveDefinite]:
    #     loc, scale = self._split_params(p)
    #
    #     if not issubclass(self.covariance_shape, Diagonal):
    #         scale_params = scale.params
    #         i_diag = (
    #             jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
    #         )
    #         scaled_params = scale_params / 2
    #         scaled_params = jnp.where(i_diag, scaled_params * 2, scaled_params)
    #         scale_params = scaled_params
    #         scale = self.covariance_shape(scale_params, self.data_dim)
    #
    #     return loc, scale
    #
    # def join_natural_params(
    #     self, loc: Array, scale: PositiveDefinite
    # ) -> Point[Natural, Normal]:
    #     scale_params = scale.params
    #     if not issubclass(self.covariance_shape, Diagonal):
    #         i_diag = (
    #             jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
    #         )
    #         scaled_params = scale_params * 2
    #         scaled_params = jnp.where(i_diag, scaled_params / 2, scaled_params)
    #         scale_params = scaled_params
    #     return Point(jnp.concatenate([loc, scale_params]))
    #


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Categorical(ClosedForm, Generative):
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

    As such, the the mapping from natural to mean parameters is given by $\\theta_i = \\log(\\eta_i/\\eta_0)$ for $i=1,\\ldots,d$.
    """

    n_categories: int

    @property
    def dimension(self) -> int:
        """Dimension $d$ is (`n_categories` - 1) due to the sum-to-one constraint."""
        return self.n_categories - 1

    # Core class methods

    def _compute_sufficient_statistic(self, x: ArrayLike) -> Array:
        one_hot = jax.nn.one_hot(x, self.n_categories)
        return one_hot[1:]

    def log_base_measure(self, x: ArrayLike) -> Array:
        return jnp.array(0.0)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        return jnp.log1p(jnp.sum(jnp.exp(natural_params)))

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        mean_params = jnp.asarray(mean_params)
        probs = self.to_probs(Point(mean_params))
        return jnp.sum(probs * jnp.log(probs))

    def sample(
        self: Categorical,
        key: ArrayLike,
        p: Point[Natural, Categorical],
        n: int = 1,
    ) -> Array:
        mean_point = self.to_mean(p)
        probs = self.to_probs(mean_point)

        key = jnp.asarray(key)
        # Use Gumbel-Max trick: argmax(log(p) + Gumbel(0,1)) ~ Categorical(p)
        g = jax.random.gumbel(key, shape=(n, self.n_categories))
        return jnp.argmax(jnp.log(probs) + g, axis=-1)

    # Additional methods

    def from_probs(self, probs: Array) -> Point[Mean, Categorical]:
        """Construct the mean parameters from the complete probabilities, dropping the first element."""
        return Point(probs[1:])

    def to_probs(self, p: Point[Mean, Categorical]) -> Array:
        """Return the probabilities of all labels."""
        probs = p.params
        prob0 = 1 - jnp.sum(probs)
        return jnp.concatenate([jnp.array([prob0]), probs])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Poisson(ClosedForm, Generative):
    """
    The Poisson distribution over counts.

    The Poisson distribution is defined by a single rate parameter $\\eta > 0$. The probability mass function at count $k \\in \\mathbb{N}$ is given by

    $$p(k; \\eta) = \\frac{\\eta^k e^{-\\eta}}{k!}.$$

    The base measure is $\\mu(k) = -\\log(k!)$, and the sufficient statistic is simply the count itself. The log-partition function is given by $\\psi(\\theta) = e^{\\theta},$ and the negative entropy is given by $\\phi(\\eta) = \\eta\\log(\\eta) - \\eta$. As such, the mapping between the natural and mean parameters is given by $\\theta = \\log(\\eta)$.
    """

    @property
    def dimension(self) -> int:
        """Single rate parameter."""
        return 1

    def _compute_sufficient_statistic(self, x: ArrayLike) -> Array:
        return jnp.asarray(x)

    def log_base_measure(self, x: ArrayLike) -> Array:
        k = jnp.asarray(x, dtype=jnp.float32)
        return -jax.lax.lgamma(k + 1)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        return jnp.squeeze(jnp.exp(jnp.asarray(natural_params)))

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        rate = jnp.asarray(mean_params)
        return jnp.squeeze(rate * (jnp.log(rate) - 1))

    def sample(
        self: Poisson, key: ArrayLike, p: Point[Natural, Poisson], n: int = 1
    ) -> Array:
        mean_point = self.to_mean(p)
        rate = mean_point.params

        key = jnp.asarray(key)
        # JAX's Poisson sampler expects rate parameter
        return jax.random.poisson(key, rate, shape=(n,))
