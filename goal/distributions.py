"""Core probability distributions as exponential families."""

from dataclasses import dataclass
from typing import Type

import jax
import jax.numpy as jnp

from goal.exponential_family import ClosedFormExponentialFamily, Parameterization
from goal.linear import Diagonal, PositiveDefinite, Scale


@dataclass(frozen=True)
class MultivariateGaussian(ClosedFormExponentialFamily):
    """Multivariate Gaussian distribution as an exponential family.

    $$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}\\exp\\left(-\frac{1}{2}(x-\\mu)^T\\Sigma^{-1}(x-\\mu)\right)$$

    Natural parameters: $\\theta = (\\theta_1, \\theta_2)$ where:
        $\\theta_1 = \\Sigma^{-1}\\mu$
        $\\theta_2 = -\frac{1}{2}\\Sigma^{-1}$
    Mean parameters: $\\eta = (\\eta_1, \\eta_2)$ where:
        $\\eta_1 = \\mu$
        $\\eta_2 = \\mu\\mu^T + \\Sigma$
    """

    data_dim: int
    covariance_shape: Type[PositiveDefinite]

    @classmethod
    def from_mean_and_covariance(
        cls, mean: jnp.ndarray, covariance: PositiveDefinite
    ) -> "MultivariateGaussian":
        return cls(
            params=jnp.concatenate([mean, covariance.params]),
            parameterization=Parameterization.SOURCE,
            data_dim=len(mean),
            covariance_shape=type(covariance),
        )

    def split_params(self) -> tuple[jnp.ndarray, PositiveDefinite]:
        operator = self.covariance_shape.from_params(
            self.params[self.data_dim :], self.data_dim
        )
        return self.params[: self.data_dim], operator

    @classmethod
    def _compute_sufficient_statistic(
        cls,
        x: jnp.ndarray,
        *,
        data_dim: int,
        covariance_shape: Type[PositiveDefinite],
        **_,
    ) -> jnp.ndarray:
        second_moment = covariance_shape.outer_product(x, x)
        return jnp.concatenate([x, second_moment.params])

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        """$$\\mu(x) = -\frac{d}{2}\\log(2\\pi)$$"""
        _, operator = self.split_params()
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        _, operator = self.split_params()
        theta1 = natural_params[: self.data_dim]
        precision = self.covariance_shape.from_params(
            natural_params[self.data_dim :], self.data_dim
        )
        x = precision.inverse() @ theta1
        return 0.25 * jnp.dot(theta1, x) - 0.5 * precision.logdet()

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        _, operator = self.split_params()
        mean = mean_params[: self.data_dim]
        second_moment = self.covariance_shape.from_params(
            mean_params[self.data_dim :], self.data_dim
        )
        outer_mean = self.covariance_shape.outer_product(mean, mean)
        covariance = second_moment - outer_mean
        return -0.5 * covariance.logdet() - 0.5 * self.data_dim * (
            1 + jnp.log(2 * jnp.pi)
        )

    def split_matrix(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return mean/location and covariance/precision as arrays.

        Returns:
            In source/mean parameterization: (mean, covariance_matrix)
            In natural parameterization: (precision_weighted_mean, rescaled_precision_matrix)
            where rescaled_precision has off-diagonal elements halved.
        """
        loc, operator = self.split_params()

        if self.parameterization == Parameterization.NATURAL:
            precision = -2.0 * operator  # theta2 = -1/2 * precision
            return precision @ loc, rescale_precision_matrix(precision)

        return loc, operator.matrix


@dataclass(frozen=True)
class Categorical(ClosedFormExponentialFamily):
    """Categorical distribution over n states as an exponential family.

    $$p(x; p) = \\prod_{i=1}^n p_i^{\\delta_{ix}}$$

    Natural parameters: $\\theta_i = \\log(p_i/p_n)$ for $i=1,\\ldots,n-1$
    Mean parameters: $\\eta_i = p_i$ for $i=1,\\ldots,n-1$
    Source parameters: $p_i$ for $i=1,\\ldots,n-1$ (same as mean)
    """

    n_categories: int

    @classmethod
    def from_source(cls, probs: jnp.ndarray) -> "Categorical":
        n_categories = len(probs) + 1
        return cls(
            n_categories=n_categories,
            params=probs,
            parameterization=Parameterization.MEAN,
        )

    @classmethod
    def from_natural(cls, log_prob_ratios: jnp.ndarray) -> "Categorical":
        n_categories = len(log_prob_ratios) + 1
        return cls(
            n_categories=n_categories,
            params=log_prob_ratios,
            parameterization=Parameterization.NATURAL,
        )

    def to_source(self) -> jnp.ndarray:
        return self.to_mean().params

    def to_probs(self) -> jnp.ndarray:
        mean_form = self.to_mean()
        probs_nm1 = mean_form.params
        last_prob = 1.0 - jnp.sum(probs_nm1)
        return jnp.concatenate([probs_nm1, jnp.array([last_prob])])

    @classmethod
    def _compute_sufficient_statistic(cls, x: jnp.ndarray, **_) -> jnp.ndarray:
        one_hot = jax.nn.one_hot(x[0], cls.n_categories)
        return one_hot[:-1]

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(0.0)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.log1p(jnp.sum(jnp.exp(natural_params)))

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        last_prob = 1.0 - jnp.sum(mean_params)
        probs = jnp.concatenate([mean_params, jnp.array([last_prob])])
        return jnp.sum(probs * jnp.log(probs))


@dataclass(frozen=True)
class Poisson(ClosedFormExponentialFamily):
    """Poisson distribution as an exponential family.

    $$p(x; \\lambda) = \frac{\\lambda^x}{x!}e^{-\\lambda}$$

    Natural parameter: $\theta = \\log(\\lambda)$
    Mean parameter: $\\eta = \\lambda$
    Source parameter: $\\lambda$ (same as mean)

    The natural/mean duality is given by:
    $$\nabla A(\theta) = \\lambda = e^\theta$$
    $$\nabla(-H(\\eta))|_{\\eta=\\lambda} = \\log(\\lambda) = \theta$$
    """

    @classmethod
    def from_source(cls, rate: float) -> "Poisson":
        return cls(params=jnp.array([rate]), parameterization=Parameterization.MEAN)

    @classmethod
    def from_natural(cls, log_rate: float) -> "Poisson":
        return cls(
            params=jnp.array([log_rate]), parameterization=Parameterization.NATURAL
        )

    def to_source(self) -> float:
        return float(self.to_mean().params[0])

    @classmethod
    def _compute_sufficient_statistic(cls, x: jnp.ndarray, **_) -> jnp.ndarray:
        return x

    def log_base_measure(self, x: jnp.ndarray) -> jnp.ndarray:
        k = x[0].astype(jnp.float32)
        return -jax.lax.lgamma(k + 1)

    def _compute_log_partition_function(
        self, natural_params: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.exp(natural_params[0])

    def _compute_negative_entropy(self, mean_params: jnp.ndarray) -> jnp.ndarray:
        rate = mean_params[0]
        return rate * (jnp.log(rate) - 1)


### Helper Functions ###


def rescale_precision_matrix(precision: PositiveDefinite) -> jnp.ndarray:
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
