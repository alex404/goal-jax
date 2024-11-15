"""Core probability distributions as exponential families."""

from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp

from goal.exponential_family import ClosedFormExponentialFamily, ParameterType
from goal.linear import create_linear_map, symmetric_to_vector, vector_to_symmetric


@dataclass(frozen=True, kw_only=True)
class MultivariateGaussian(ClosedFormExponentialFamily):
    """
    Multivariate Gaussian distribution with flexible covariance representation.

    Source parameters: (μ, Σ) mean and covariance
    Natural parameters: [η, vec(Θ)] where η = Σ^{-1}μ, Θ = -1/2 Σ^{-1}
    Mean parameters: [μ, vec(M)] where M = μμ^T + Σ
    """

    dim: int
    operator_type: str  # 'general', 'symmetric', 'posdef', 'diagonal', or 'scale'
    params: jnp.ndarray
    _param_type: ParameterType

    @staticmethod
    def parameter_count(dim: int, operator_type: str) -> int:
        """Return number of parameters needed for given dimension and operator type."""
        op = create_linear_map(jnp.eye(dim), operator_type)
        return dim + op.parameter_count()

    @classmethod
    def from_source(
        cls,
        mean: jnp.ndarray,
        covariance: Union[jnp.ndarray, float],
        operator_type: str = "posdef",
    ) -> "MultivariateGaussian":
        """Create from source parameters with specified operator type."""
        if operator_type not in ["general", "symmetric", "posdef", "diagonal", "scale"]:
            raise ValueError(f"Invalid operator type: {operator_type}")

        dim = mean.shape[0]
        operator = create_linear_map(covariance, operator_type)

        # Compute natural parameters
        precision = operator.inverse()
        eta = precision @ mean
        theta = -0.5 * precision.to_vector()
        natural_params = jnp.concatenate([eta, theta])

        expected_size = cls.parameter_count(dim, operator_type)
        if len(natural_params) != expected_size:
            raise ValueError(
                f"Parameter dimension mismatch. Expected {expected_size}, got {len(natural_params)}"
            )

        return cls(
            dim=dim,
            params=natural_params,
            _param_type=ParameterType.NATURAL,
            operator_type=operator_type,
        )

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns flat array of [x, vec(xx^T)]."""
        if x.shape != (self.dim,):
            raise ValueError(
                f"Input shape {x.shape} does not match dimension {self.dim}"
            )

        outer_x = jnp.outer(x, x)
        if self.operator_type in ["general", "symmetric", "posdef"]:
            second_moment = symmetric_to_vector(outer_x)
        elif self.operator_type == "diagonal":
            second_moment = jnp.diag(outer_x)
        else:  # scale
            second_moment = jnp.array([jnp.mean(jnp.diag(outer_x))])

        return jnp.concatenate([x, second_moment])

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log base measure (constant)."""
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log partition function for natural parameters."""
        eta = natural_params[: self.dim]
        theta_vec = natural_params[self.dim :]
        precision = create_linear_map(-2 * theta_vec, self.operator_type)
        x = precision.inverse() @ eta
        logdet = precision.logdet()
        return 0.25 * jnp.dot(eta, x) - 0.5 * logdet

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        """Compute negative entropy for mean parameters."""
        mean = mean_params[: self.dim]
        if self.operator_type in ["general", "symmetric", "posdef"]:
            second_moment = vector_to_symmetric(mean_params[self.dim :], self.dim)
        elif self.operator_type == "diagonal":
            second_moment = jnp.diag(mean_params[self.dim :])
        else:  # scale
            second_moment = mean_params[self.dim] * jnp.eye(self.dim)

        covariance = second_moment - jnp.outer(mean, mean)
        cov_op = create_linear_map(covariance, self.operator_type)
        return -0.5 * cov_op.logdet() - 0.5 * self.dim * (1 + jnp.log(2 * jnp.pi))


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
    _param_type: ParameterType

    @classmethod
    def from_source(cls, probs: jnp.ndarray) -> "Categorical":
        """Create from first n-1 probabilities (source parameters)."""
        if len(probs) < 1:
            raise ValueError("Must have at least 2 categories")

        n_categories = len(probs) + 1
        return cls(
            n_categories=n_categories, params=probs, _param_type=ParameterType.MEAN
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
            _param_type=ParameterType.NATURAL,
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
    _param_type: ParameterType

    @classmethod
    def from_source(cls, rate: float) -> "Poisson":
        """Create from rate parameter λ (source parameter)."""
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        return cls(params=jnp.array([rate]), _param_type=ParameterType.MEAN)

    @classmethod
    def from_natural(cls, log_rate: float) -> "Poisson":
        """Create from log rate parameter log(λ)."""
        return cls(params=jnp.array([log_rate]), _param_type=ParameterType.NATURAL)

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
