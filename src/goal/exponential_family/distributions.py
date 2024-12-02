"""Core exponential families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from goal.manifold import Coordinates, Euclidean, Point, reduce_double_dual

from ..exponential_family import (
    ClosedForm,
    Generative,
    Mean,
    Natural,
)
from ..linear import (
    Diagonal,
    Identity,
    PositiveDefinite,
    Scale,
    SquareMap,
)

type FullNormal = Normal[PositiveDefinite]
type DiagonalNormal = Normal[Diagonal]
type IsotropicNormal = Normal[Scale]
type StandardNormal = Normal[Identity]

type Covariance[R: PositiveDefinite] = SquareMap[R, Euclidean]
type FullCovariance = Covariance[PositiveDefinite]
type DiagonalCovariance = Covariance[Diagonal]
type IsotropicCovariance = Covariance[Scale]


def full_normal_manifold(
    ndim: int = 1,
) -> FullNormal:
    """Create an unconstrained Normal distribution.

    Args:
        ndim: Dimension of the space
    """
    return Normal(SquareMap(PositiveDefinite(), Euclidean(ndim)))


def diagonal_normal_manifold(
    ndim: int = 1,
) -> DiagonalNormal:
    """Create a diagonal-constrained Normal distribution.

    Args:
        ndim: Dimension of the space
    """
    return Normal(SquareMap(Diagonal(), Euclidean(ndim)))


def isotropic_normal_manifold(
    ndim: int = 1,
) -> IsotropicNormal:
    """Create an isotropic-constrained Normal distribution.

    Args:
        ndim: Dimension of the space
    """
    return Normal(SquareMap(Scale(), Euclidean(ndim)))


def standard_normal_manifold(
    ndim: int = 1,
) -> StandardNormal:
    """Create a (0-Dimensional) manifold of standard Normal distributions.

    Args:
        ndim: Dimension of the space
    """
    return Normal(SquareMap(Identity(), Euclidean(ndim)))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Normal[R: PositiveDefinite](ClosedForm, Generative):
    """(Multivariate) Normal distributions.

    Parameters:
        cov_man: Covariance structure class (e.g. `Scale`, `Diagonal`, `PositiveDefinite`). Determines how the covariance matrix is parameterized.

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

    cov_man: Covariance[R]
    """Covariance structure (e.g. `Scale`, `Diagonal`, `PositiveDefinite`)."""

    @property
    def data_dimension(self) -> int:
        """Dimension of the data space."""
        return self.cov_man.shape[0]

    @property
    def dimension(self) -> int:
        """Total dimension = `data_dim` + `covariance_dim`."""
        return self.data_dimension + self.cov_man.dimension

    # Core class methods

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)

        # Create point in StandardNormal
        x_point: Point[Mean, Euclidean] = Point(x)

        # Compute outer product with appropriate structure
        second_moment: Point[Mean, Covariance[R]] = self.cov_man.outer_product(
            x_point, x_point
        )

        # Concatenate components into parameter vector
        return self._join_params(x_point, second_moment)

    def log_base_measure(self, x: Array) -> Array:
        return -0.5 * self.data_dimension * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        natural_params = jnp.asarray(natural_params)
        theta1, theta2 = self.split_natural_params(Point(natural_params))

        precision = -2 * theta2
        covariance: Point[Mean, Covariance[R]] = reduce_double_dual(
            self.cov_man.inverse(precision)
        )
        mean = self.cov_man(covariance, theta1)

        return 0.5 * jnp.dot(theta1.params, mean.params) - 0.5 * self.cov_man.logdet(
            precision
        )

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        mean_params = jnp.asarray(mean_params)
        mean, second_moment = self.split_mean_params(Point(mean_params))

        # Compute covariance without forming full matrices
        outer_mean = self.cov_man.outer_product(mean, mean)
        covariance = second_moment - outer_mean

        log_det = self.cov_man.logdet(covariance)
        entropy = 0.5 * (
            self.data_dimension + self.data_dimension * jnp.log(2 * jnp.pi) + log_det
        )

        return -entropy

    def sample(
        self,
        key: ArrayLike,
        p: Point[Natural, Self],
        n: int = 1,
    ) -> Array:
        mean_point = self.to_mean(p)
        mean, covariance = self.to_mean_and_covariance(mean_point)

        # Draw standard normal samples
        key = jnp.asarray(key)
        shape = (n, self.data_dimension)
        z = jax.random.normal(key, shape)

        # Transform samples using Cholesky
        samples = self.cov_man.rep.apply_cholesky(
            covariance.params, z, self.cov_man.shape
        )
        return mean.params + samples

    # Additional methods

    def _split_params[C: Coordinates](
        self, p: Point[C, Self]
    ) -> tuple[Point[C, Euclidean], Point[C, Covariance[R]]]:
        """Split parameters into location and scale components.

        Args:
            p: Point in Normal manifold coordinates

        Returns:
            Tuple of (location vector, covariance matrix)
        """
        loc_params = p.params[: self.data_dimension]
        cov_params = p.params[self.data_dimension :]

        return Point(loc_params), Point(cov_params)

    def _join_params[C: Coordinates](
        self,
        loc: Point[C, Euclidean],
        cov: Point[C, Covariance[R]],
    ) -> Array:
        """Join location and scale parameters into a single array."""
        return jnp.concatenate([loc.params, cov.params])

    def from_mean_and_covariance(
        self,
        mean: Point[Mean, Euclidean],
        covariance: Point[Mean, Covariance[R]],
    ) -> Point[Mean, Self]:
        """Construct a `Point` in `Mean` coordinates from the mean $\\mu$ and covariance $\\Sigma$."""
        # Create the second moment η₂ = μμᵀ + Σ
        outer = self.cov_man.outer_product(mean, mean)
        second_moment = outer + covariance

        return Point(self._join_params(mean, second_moment))

    def to_mean_and_covariance(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, Covariance[R]]]:
        """Extract the mean $\\mu$ and covariance $\\Sigma$ from a `Point` in `Mean` coordinates."""
        # Split into μ and η₂
        mean, second_moment = self._split_params(p)

        # Compute Σ = η₂ - μμᵀ
        outer = self.cov_man.outer_product(mean, mean)
        covariance = second_moment - outer

        return mean, covariance

    def split_mean_params(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, Covariance[R]]]:
        """Split parameters into mean and second-moment components."""
        return self._split_params(p)

    def join_mean_params(
        self, mean: Point[Mean, Euclidean], second_moment: Point[Mean, Covariance[R]]
    ) -> Point[Mean, Self]:
        """Join mean and second-moment parameters."""
        return Point(self._join_params(mean, second_moment))

    def split_natural_params(
        self, p: Point[Natural, Self]
    ) -> tuple[Point[Natural, Euclidean], Point[Natural, Covariance[R]]]:
        """Split parameters into natural location and precision components."""
        # First do basic parameter split
        loc, precision = self._split_params(p)

        # We need to rescale off-precision params
        if not isinstance(self.cov_man.rep, Diagonal):
            precision_params = precision.params
            i_diag = (
                jnp.triu_indices(self.data_dimension)[0]
                == jnp.triu_indices(self.data_dimension)[1]
            )

            scaled_params = precision_params / 2  # First undo the -1/2
            scaled_params = jnp.where(i_diag, scaled_params * 2, scaled_params)
            precision: Point[Natural, Covariance[R]] = Point(scaled_params)

        return loc, precision

    def join_natural_params(
        self,
        loc: Point[Natural, Euclidean],
        precision: Point[Natural, Covariance[R]],
    ) -> Point[Natural, Self]:
        """Join natural location and precision parameters."""
        precision_params = precision.params

        # We need to rescale off-precision params
        if not isinstance(self.cov_man.rep, Diagonal):
            i_diag = (
                jnp.triu_indices(self.data_dimension)[0]
                == jnp.triu_indices(self.data_dimension)[1]
            )

            scaled_params = precision_params * 2  # First multiply by -1/2
            scaled_params = jnp.where(i_diag, scaled_params / 2, scaled_params)
            precision_params = scaled_params

        return Point(self._join_params(loc, Point(precision_params)))


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

    @property
    def data_dimension(self) -> int:
        """Dimension of the data space."""
        return 1

    # Core class methods

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        one_hot = jax.nn.one_hot(x, self.n_categories)
        return one_hot[1:]

    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        return jnp.log1p(jnp.sum(jnp.exp(natural_params)))

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        mean_params = jnp.asarray(mean_params)
        probs = self.to_probs(Point(mean_params))
        return jnp.sum(probs * jnp.log(probs))

    def sample(
        self,
        key: ArrayLike,
        p: Point[Natural, Self],
        n: int = 1,
    ) -> Array:
        mean_point = self.to_mean(p)
        probs = self.to_probs(mean_point)

        key = jnp.asarray(key)
        # Use Gumbel-Max trick: argmax(log(p) + Gumbel(0,1)) ~ Categorical(p)
        g = jax.random.gumbel(key, shape=(n, self.n_categories))
        return jnp.argmax(jnp.log(probs) + g, axis=-1)

    # Additional methods

    def from_probs(self, probs: Array) -> Point[Mean, Self]:
        """Construct the mean parameters from the complete probabilities, dropping the first element."""
        return Point(probs[1:])

    def to_probs(self, p: Point[Mean, Self]) -> Array:
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

    @property
    def data_dimension(self) -> int:
        return 1

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        return jnp.asarray(x)

    def log_base_measure(self, x: Array) -> Array:
        k = jnp.asarray(x, dtype=jnp.float32)
        return -jax.lax.lgamma(k + 1)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        return jnp.squeeze(jnp.exp(jnp.asarray(natural_params)))

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        rate = jnp.asarray(mean_params)
        return jnp.squeeze(rate * (jnp.log(rate) - 1))

    def sample(self, key: ArrayLike, p: Point[Natural, Self], n: int = 1) -> Array:
        mean_point = self.to_mean(p)
        rate = mean_point.params

        key = jnp.asarray(key)
        # JAX's Poisson sampler expects rate parameter
        return jax.random.poisson(key, rate, shape=(n,))