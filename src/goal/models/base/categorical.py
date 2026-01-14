"""Categorical distributions over finite sets.

This module provides:
- `Categorical`: Distribution over n states with probabilities summing to 1
- `Bernoulli`: Special case for binary variables (equivalent to Categorical(2))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import Analytic


@dataclass(frozen=True)
class Bernoulli(Analytic):
    """Bernoulli distribution for a single binary variable.

    Mathematically equivalent to Categorical(n_categories=2).

    The distribution over binary values x ∈ {0, 1} is:

    .. math::

        p(x; \\theta) = \\sigma(\\theta)^x (1 - \\sigma(\\theta))^{1-x}

    where σ(θ) = 1/(1 + exp(-θ)) is the sigmoid function.

    As an exponential family:
        - Sufficient statistic: s(x) = x (identity)
        - Base measure: μ(x) = 0
        - Natural parameter: θ = log(p/(1-p)) (log odds)
        - Mean parameter: η = p = P(x=1)
        - Log partition: ψ(θ) = log(1 + exp(θ)) = softplus(θ)
        - Negative entropy: φ(η) = η*log(η) + (1-η)*log(1-η)
    """

    @property
    @override
    def dim(self) -> int:
        """Parameter dimension is 1."""
        return 1

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension is 1 (single binary value)."""
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Identity sufficient statistic s(x) = x."""
        return jnp.atleast_1d(x).astype(jnp.float32)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure is constant (zero in log space)."""
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Log partition function: log(1 + exp(θ)) = softplus(θ)."""
        return jax.nn.softplus(params[0])

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Negative entropy: η*log(η) + (1-η)*log(1-η)."""
        p = means[0]
        p0 = 1 - p
        # Add small epsilon for numerical stability
        eps = 1e-10
        return p * jnp.log(p + eps) + p0 * jnp.log(p0 + eps)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from Bernoulli distribution.

        Args:
            key: JAX random key
            params: Natural parameters (log odds)
            n: Number of samples

        Returns:
            Array of shape (n, 1) with binary values
        """
        prob = jax.nn.sigmoid(params[0])
        return jax.random.bernoulli(key, prob, shape=(n, 1)).astype(jnp.float32)

    # Convenience methods

    def to_prob(self, means: Array) -> Array:
        """Extract P(x=1) from mean parameters."""
        return means[0]

    def from_prob(self, prob: float) -> Array:
        """Construct mean parameters from P(x=1)."""
        return jnp.array([prob])


@dataclass(frozen=True)
class Categorical(Analytic):
    """Categorical distribution over $n$ states.

    The categorical distribution describes discrete probability distributions over $n$ states with probabilities $\\eta_i$ where $\\sum_{i=0}^n \\eta_i = 1$.

    $$p(k; \\eta) = \\eta_k$$

    As an exponential family:

    - Base measure: $\\mu(k) = 0$
    - Sufficient statistic: One-hot encoding for $k > 0$
    - Log partition: $\\psi(\\theta) = \\log(1 + \\sum_{i=1}^d e^{\\theta_i})$
    - Negative entropy: $\\phi(\\eta) = \\sum_{i=0}^d \\eta_i \\log(\\eta_i)$
    """

    n_categories: int
    """Number of categories."""

    @property
    @override
    def dim(self) -> int:
        """Dimension $d$ is `n_categories - 1` due to the sum-to-one constraint."""
        return self.n_categories - 1

    @property
    @override
    def data_dim(self) -> int:
        """Dimension of the data space."""
        return 1

    # Categorical methods

    def from_probs(self, probs: Array) -> Array:
        """Construct the mean parameters from the complete probabilities, dropping the first element."""
        return probs[1:]

    def to_probs(self, means: Array) -> Array:
        """Return the probabilities of all labels."""
        prob0 = 1 - jnp.sum(means)
        return jnp.concatenate([jnp.array([prob0]), means])

    # Overrides

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return jax.nn.one_hot(x - 1, self.n_categories - 1).reshape(-1)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        array = jnp.concatenate([jnp.array([0.0]), params])
        max_val = jnp.max(array)
        return max_val + jax.nn.logsumexp(array - max_val)

    @override
    def negative_entropy(self, means: Array) -> Array:
        probs = self.to_probs(means)
        return jnp.sum(probs * jnp.log(probs))

    @override
    def sample(
        self,
        key: Array,
        params: Array,
        n: int = 1,
    ) -> Array:
        means = self.to_mean(params)
        probs = self.to_probs(means)

        key = jnp.asarray(key)
        # Use Gumbel-Max trick: argmax(log(p) + Gumbel(0,1)) ~ Categorical(p)
        g = jax.random.gumbel(key, shape=(n, self.n_categories))
        return jnp.argmax(jnp.log(probs) + g, axis=-1)[..., None]
