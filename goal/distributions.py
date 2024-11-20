"""Core probability distributions as exponential families."""

from dataclasses import dataclass
from typing import Type

import jax
import jax.numpy as jnp

from goal.exponential_family import ClosedFormExponentialFamily, Mean, Source
from goal.linear import Diagonal, PositiveDefinite, Scale
from goal.manifold import Point


@dataclass(frozen=True)
class MultivariateGaussian(ClosedFormExponentialFamily):
    """Multivariate Gaussian distribution as an exponential family.

    $$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}\\exp\\left(-\\frac{1}{2}(x-\\mu)^T\\Sigma^{-1}(x-\\mu)\\right)$$

    The parameter mappings between coordinates are:

    Natural ↔ Source/Mean
    * $\\theta_1 = \\Sigma^{-1}\\mu$
    * $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$
    * $\\eta_1 = \\mu$
    * $\\eta_2 = \\mu\\mu^T + \\Sigma$

    Different covariance structures (full, diagonal, scalar) are handled efficiently
    through the LinearOperator hierarchy.
    """

    data_dim: int
    covariance_shape: Type[PositiveDefinite]

    @property
    def dim(self) -> int:
        """Total dimension = mean_dim + covariance_params."""
        return self.data_dim + self.covariance_dim

    @property
    def covariance_dim(self) -> int:
        """Number of parameters needed for covariance structure."""
        if issubclass(self.covariance_shape, Scale):
            return 1
        if issubclass(self.covariance_shape, Diagonal):
            return self.data_dim
        if issubclass(self.covariance_shape, PositiveDefinite):
            n = self.data_dim
            return (n * (n + 1)) // 2  # Triangular number for symmetric matrix
        raise ValueError(f"Unknown covariance shape: {self.covariance_shape}")

    def from_mean_and_covariance(
        self, mean: jnp.ndarray, covariance: PositiveDefinite
    ) -> Point[Source, "MultivariateGaussian"]:
        """Construct point from standard mean and covariance parameters."""
        params = jnp.concatenate([mean, covariance.params])
        return Point(params, self)

    def split_params(self, p: Point) -> tuple[jnp.ndarray, PositiveDefinite]:
        """Split parameters into location and scale components.

        The interpretation depends on the coordinate system:
        - Natural: (Σ⁻¹μ, -1/2 Σ⁻¹)
        - Mean: (μ, μμᵀ + Σ)
        - Source: (μ, Σ)
        """
        operator = self.covariance_shape.from_params(
            p.params[self.data_dim :], self.data_dim
        )
        return p.params[: self.data_dim], operator

    def _compute_sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map observation to sufficient statistics (x, xxᵀ)."""
        second_moment = self.covariance_shape.outer_product(x, x)
        return jnp.concatenate([x, second_moment.params])

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Base measure $$\\mu(x) = -\\frac{d}{2}\\log(2\\pi)$$"""
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log partition function.

        $$\\psi(\\theta) = \\frac{1}{4}\\theta_1^T\\Sigma\\theta_1 - \\frac{1}{2}\\log|\\Sigma^{-1}|$$

        where $\\Sigma^{-1} = -2\\theta_2$.
        """
        theta1 = natural_params[: self.data_dim]
        precision = self.covariance_shape.from_params(
            natural_params[self.data_dim :], self.data_dim
        )

        # Convert from natural params: theta2 = -1/2 Σ⁻¹
        precision = -2.0 * precision
        covariance = precision.inverse()

        # Compute Σθ₁ without forming full matrices
        mean = covariance.matvec(theta1)
        quad_term = jnp.dot(theta1, mean)

        return 0.25 * quad_term - 0.5 * precision.logdet()

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute negative entropy.

        $$-H(\\eta) = \\frac{1}{2}\\log|\\Sigma| + \\frac{d}{2}(1 + \\log(2\\pi))$$

        where $\\Sigma = \\eta_2 - \\eta_1\\eta_1^T$ is the covariance.
        """
        mean, second_moment = self.split_params(Point(mean_params, self))

        # Compute covariance without forming full matrices
        outer_mean = self.covariance_shape.outer_product(mean, mean)
        covariance = second_moment - outer_mean

        return -0.5 * covariance.logdet() - 0.5 * self.data_dim * (
            1 + jnp.log(2 * jnp.pi)
        )


@dataclass(frozen=True)
class Categorical(ClosedFormExponentialFamily):
    """Categorical distribution over n states as an exponential family.

    $$p(x; p) = \\prod_{i=1}^n p_i^{\\delta_{ix}}$$

    The parameter mappings between coordinates are:
    Natural ↔ Source/Mean:
        * $\\theta_i = \\log(p_i/p_n)$ for $i=1,\\ldots,n-1$
        * $\\eta_i = p_i$ for $i=1,\\ldots,n-1$
        * Source parameters are the same as mean parameters

    Note: The nth probability is determined by the constraint $\\sum_i p_i = 1$
    """

    n_categories: int

    @property
    def dim(self) -> int:
        """Dimension is n-1 due to sum-to-one constraint."""
        return self.n_categories - 1

    def from_probs(self, probs: jnp.ndarray) -> Point[Source, "Categorical"]:
        """Construct from probability vector (last entry determined by constraint)."""
        return Point(probs[:-1], self)  # Drop last prob due to constraint

    def to_probs(self, p: Point[Mean, "Categorical"]) -> jnp.ndarray:
        """Convert to full probability vector."""
        probs_nm1 = p.params
        last_prob = 1.0 - jnp.sum(probs_nm1)
        return jnp.concatenate([probs_nm1, jnp.array([last_prob])])

    def _compute_sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map observation to one-hot encoding (dropping last component)."""
        one_hot = jax.nn.one_hot(x[0], self.n_categories)
        return one_hot[:-1]

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Base measure is constant."""
        return jnp.array(0.0)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute $$\\psi(\\theta) = \\log(1 + \\sum_{i=1}^{n-1} e^{\\theta_i})$$"""
        return jnp.log1p(jnp.sum(jnp.exp(natural_params)))

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute $$-H(\\eta) = \\sum_{i=1}^n p_i \\log p_i$$"""
        probs = self.to_probs(Point(mean_params, self))
        return jnp.sum(probs * jnp.log(probs))


@dataclass(frozen=True)
class Poisson(ClosedFormExponentialFamily):
    """Poisson distribution as an exponential family.

    $$p(x; \\lambda) = \\frac{\\lambda^x}{x!}e^{-\\lambda}$$

    The parameter mappings between coordinates are:
    Natural ↔ Source/Mean:
        * $\\theta = \\log(\\lambda)$
        * $\\eta = \\lambda$
        * Source parameters are the same as mean parameters

    The natural/mean duality is given by:
    * $\\nabla \\psi(\\theta) = \\lambda = e^\\theta$
    * $\\nabla(-H(\\eta))|_{\\eta=\\lambda} = \\log(\\lambda) = \\theta$
    """

    @property
    def dim(self) -> int:
        """Single rate parameter."""
        return 1

    def from_rate(self, rate: float) -> Point[Source, "Poisson"]:
        """Construct from rate parameter λ."""
        return Point(jnp.array([rate]), self)

    def _compute_sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Sufficient statistic is the count itself."""
        return x

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Base measure $$\\mu(x) = -\\log(x!)$$"""
        k = x[0].astype(jnp.float32)
        return -jax.lax.lgamma(k + 1)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute $$\\psi(\\theta) = e^\\theta$$"""
        return jnp.exp(natural_params[0])

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute $$-H(\\eta) = \\eta\\log(\\eta) - \\eta$$"""
        rate = mean_params[0]
        return rate * (jnp.log(rate) - 1)
