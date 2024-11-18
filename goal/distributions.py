"""Core probability distributions as exponential families."""

from dataclasses import dataclass, replace
from typing import Protocol, Type

import jax
import jax.numpy as jnp

from goal.exponential_family import ClosedFormExponentialFamily, Parameterization
from goal.linear import Diagonal, LinearOperator, Scale, Symmetric


class CovarianceMatrix(Protocol):
    """Protocol for covariance matrices in exponential families.

    Requires methods needed for efficient computation of exponential
    family operations like log partition functions and entropy.
    """

    def matvec(self, v: jnp.ndarray) -> jnp.ndarray: ...
    def transpose(self) -> "CovarianceMatrix": ...
    def logdet(self) -> jnp.ndarray: ...
    def inverse(self) -> "CovarianceMatrix": ...
    def to_vector(self) -> jnp.ndarray: ...
    @property
    def cholesky(self) -> jnp.ndarray: ...


@dataclass(frozen=True)
class MultivariateGaussian(ClosedFormExponentialFamily):
    """Multivariate Gaussian distribution.

    The multivariate Gaussian distribution as an exponential family:

    $$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}\\exp\\left(-\\frac{1}{2}(x-\\mu)^T\\Sigma^{-1}(x-\\mu)\\right)$$

    Natural parameters: $\\theta = (\\theta_1, \\theta_2)$ where:
        $\\theta_1 = \\Sigma^{-1}\\mu$
        $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$
    Mean parameters: $\\eta = (\\eta_1, \\eta_2)$ where:
        $\\eta_1 = \\mu$
        $\\eta_2 = \\mu\\mu^T + \\Sigma$
    """

    dim: int
    params: jnp.ndarray
    _param_type: Parameterization
    _operator_type: Type[LinearOperator]

    @classmethod
    def from_mean_and_covariance(
        cls, mean: jnp.ndarray, covariance: LinearOperator
    ) -> "MultivariateGaussian":
        return cls(
            dim=mean.shape[0],
            params=jnp.concatenate([mean, covariance.params]),
            _param_type=Parameterization.SOURCE,
            _operator_type=type(covariance),
        )

    def split_params(self) -> tuple[jnp.ndarray, LinearOperator]:
        """Split parameters into vector and operator components."""
        loc, scale_vec = self.params[: self.dim], self.params[self.dim :]
        return loc, self._operator_type.from_matrix(scale_vec)

    def split_vector_matrix(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split parameters into vector and full matrix components."""
        loc, operator = self.split_params()
        if self._param_type == Parameterization.NATURAL:
            return loc, rescale_precision_matrix(operator)
        return loc, operator.matrix

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns sufficient statistics $[x, \\text{vec}(xx^T)]$."""
        second_moment = jnp.outer(x, x)
        sym = Symmetric.from_matrix(second_moment)
        return jnp.concatenate([x, sym.params])

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log base measure.

        $$h(x) = -\\frac{d}{2}\\log(2\\pi)$$
        """
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        theta1, precision = natural_params[: self.dim], natural_params[self.dim :]
        precision_op = self._operator_type.from_matrix(precision)
        x = precision_op.inverse() @ theta1
        return 0.25 * jnp.dot(theta1, x) - 0.5 * precision_op.logdet()

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        mean, second_moment = mean_params[: self.dim], mean_params[self.dim :]
        second_moment_op = self._operator_type.from_matrix(second_moment)
        covariance_op = second_moment_op - self._operator_type.outer_product(mean, mean)
        return -0.5 * covariance_op.logdet() - 0.5 * self.dim * (
            1 + jnp.log(2 * jnp.pi)
        )

    def to_mean(self) -> "MultivariateGaussian":
        """Convert to mean parameters."""
        if self._param_type == Parameterization.SOURCE:
            mu, sigma = self.split_params()
            second_moment = sigma.matrix + jnp.outer(mu, mu)
            return replace(
                self,
                params=jnp.concatenate([mu, second_moment]),
                _param_type=Parameterization.MEAN,
            )

        # If not source, rely on superclass method
        return super().to_mean()

    def to_natural(self) -> "MultivariateGaussian":
        """Convert to natural parameters."""
        if self._param_type == Parameterization.SOURCE:
            self.to_mean().to_natural()

        return super().to_natural()


@dataclass(frozen=True, kw_only=True)
class Categorical(ClosedFormExponentialFamily):
    """
    Categorical distribution over n states, parameterized by n-1 parameters.

    Source parameters: p[:-1] first n-1 probabilities (equivalent to mean parameters)
    Natural parameters: log(p[:-1]/p[-1])
    Mean parameters: p[:-1]
    """

    n_categories: int
    params: jnp.ndarray
    _param_type: Parameterization

    @classmethod
    def from_source(cls, probs: jnp.ndarray) -> "Categorical":
        """Create from first n-1 probabilities (source parameters)."""
        if len(probs) < 1:
            raise ValueError("Must have at least 2 categories")

        n_categories = len(probs) + 1
        return cls(
            n_categories=n_categories, params=probs, _param_type=Parameterization.MEAN
        )

    @classmethod
    def from_natural(cls, log_prob_ratios: jnp.ndarray) -> "Categorical":
        """Create from log probability ratios log(p[:-1]/p[-1])."""
        if len(log_prob_ratios) < 1:
            raise ValueError("Must have at least 2 categories")

        n_categories = len(log_prob_ratios) + 1
        return cls(
            n_categories=n_categories,
            params=log_prob_ratios,
            _param_type=Parameterization.NATURAL,
        )

    def to_source(self) -> jnp.ndarray:
        """Convert to source parameters (first n-1 probabilities)."""
        return self.to_mean().params

    def to_probs(self) -> jnp.ndarray:
        """Convert to full probability vector."""
        mean_form = self.to_mean()
        probs_nm1 = mean_form.params
        last_prob = 1.0 - jnp.sum(probs_nm1)
        return jnp.concatenate([probs_nm1, jnp.array([last_prob])])

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """One-hot encoding minus last category."""
        if x.shape != (1,) or not (0 <= x[0] < self.n_categories):
            raise ValueError(
                f"Invalid input {x} for categorical with {self.n_categories} categories"
            )
        one_hot = jax.nn.one_hot(x[0], self.n_categories)
        return one_hot[:-1]

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Log base measure (constant)."""
        return jnp.array(0.0)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Log partition function is log(1 + sum(exp(natural_params)))."""
        return jnp.log1p(jnp.sum(jnp.exp(natural_params)))

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Negative entropy for categorical distribution."""
        last_prob = 1.0 - jnp.sum(mean_params)
        probs = jnp.concatenate([mean_params, jnp.array([last_prob])])
        return jnp.sum(probs * jnp.log(probs))


@dataclass(frozen=True, kw_only=True)
class Poisson(ClosedFormExponentialFamily):
    """
    Poisson distribution.

    Source parameter: λ rate parameter (equivalent to mean parameter)
    Natural parameter: log(λ)
    Mean parameter: λ

    Note that the derivatives of negative entropy and log partition function are inverses:
    ∇A(η) = λ = exp(η)
    ∇(-H(μ))|_{μ=λ} = log(λ) = η
    """

    params: jnp.ndarray
    _param_type: Parameterization

    @classmethod
    def from_source(cls, rate: float) -> "Poisson":
        """Create from rate parameter λ (source parameter)."""
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        return cls(params=jnp.array([rate]), _param_type=Parameterization.MEAN)

    @classmethod
    def from_natural(cls, log_rate: float) -> "Poisson":
        """Create from log rate parameter log(λ)."""
        return cls(params=jnp.array([log_rate]), _param_type=Parameterization.NATURAL)

    def to_source(self) -> float:
        """Get source parameter (rate λ)."""
        return float(self.to_mean().params[0])

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Sufficient statistic is x."""
        if x.shape != (1,) or x[0] < 0:
            raise ValueError(f"Input must be non-negative scalar, got {x}")
        return x

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Log base measure is -log(x!)."""
        # Convert to float32 for lgamma compatibility
        k = x[0].astype(jnp.float32)
        return -jax.lax.lgamma(k + 1)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Log partition function is exp(η)."""
        return jnp.exp(natural_params[0])

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Negative entropy is λ(log(λ) - 1)."""
        rate = mean_params[0]
        return rate * (jnp.log(rate) - 1)


### Helper Functions ###


def rescale_precision_matrix(precision: LinearOperator) -> jnp.ndarray:
    """Rescale precision matrix for natural parameters.

    For symmetric matrices with non-zero off-diagonal elements, halves those elements
    to account for the fact that e.g. the interactions $x_1x_2$ and $x_2x_1$ are counted
    only once in the minimal representation of natural parameters, but both are present
    when using the precision matrix as a quadratic form.
    """
    if isinstance(precision, (Scale, Diagonal)):
        return precision.matrix  # No off-diagonals to scale

    matrix = precision.matrix
    off_diag_mask = ~jnp.eye(matrix.shape[0], dtype=bool)
    return matrix.at[off_diag_mask].divide(2)
