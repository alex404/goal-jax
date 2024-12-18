"""Core exponential families. In particular provides

- `Categorical`: Family of discrete probability distributions over $n$ states.
- `Poisson`: Family of Poisson distribution over counts $k \\in \\mathbb{N}$.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from ..geometry import Backward, Mean, Natural, Point


@dataclass(frozen=True)
class Categorical(Backward):
    """Categorical distribution over $n$ states.

    The categorical distribution describes discrete probability distributions over $n$ states with probabilities $\\eta_i$ where $\\sum_{i=0}^n \\eta_i = 1$. The probability mass function for outcome $k$ is:

    $$p(k; \\eta) = \\eta_k$$

    Properties:

    - Base measure: $\\mu(k) = 0$
    - Sufficient statistic: One-hot encoding for $k > 0$
    - Log partition: $\\psi(\\theta) = \\log(1 + \\sum_{i=1}^d e^{\\theta_i})$
    - Negative entropy: $\\phi(\\eta) = \\sum_{i=0}^d \\eta_i \\log(\\eta_i)$
    """

    n_categories: int

    @property
    def dim(self) -> int:
        """Dimension $d$ is (`n_categories` - 1) due to the sum-to-one constraint."""
        return self.n_categories - 1

    @property
    def data_dim(self) -> int:
        """Dimension of the data space."""
        return 1

    # Core class methods

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        return jax.nn.one_hot(x - 1, self.n_categories - 1).reshape(-1)

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
        key: Array,
        p: Point[Natural, Self],
        n: int = 1,
    ) -> Array:
        mean_point = self.to_mean(p)
        probs = self.to_probs(mean_point)

        key = jnp.asarray(key)
        # Use Gumbel-Max trick: argmax(log(p) + Gumbel(0,1)) ~ Categorical(p)
        g = jax.random.gumbel(key, shape=(n, self.n_categories))
        return jnp.argmax(jnp.log(probs) + g, axis=-1)[..., None]

    # Additional methods

    def from_probs(self, probs: Array) -> Point[Mean, Self]:
        """Construct the mean parameters from the complete probabilities, dropping the first element."""
        return Point(probs[1:])

    def to_probs(self, p: Point[Mean, Self]) -> Array:
        """Return the probabilities of all labels."""
        probs = p.params
        prob0 = 1 - jnp.sum(probs)
        return jnp.concatenate([jnp.array([prob0]), probs])


@dataclass(frozen=True)
class Poisson(Backward):
    """
    The Poisson distribution over counts.

    The Poisson distribution is defined by a single rate parameter $\\eta > 0$. The probability mass function at count $k \\in \\mathbb{N}$ is given by

    $$p(k; \\eta) = \\frac{\\eta^k e^{-\\eta}}{k!}.$$

    Properties:

    - Base measure $\\mu(k) = -\\log(k!)$
    - Sufficient statistic $s(x) = x$
    - Log-partition function: $\\psi(\\theta) = e^{\\theta}$
    - Negative entropy: $\\phi(\\eta) = \\eta\\log(\\eta) - \\eta$
    """

    @property
    def dim(self) -> int:
        """Single rate parameter."""
        return 1

    @property
    def data_dim(self) -> int:
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

    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        mean_point = self.to_mean(p)
        rate = mean_point.params

        key = jnp.asarray(key)
        # JAX's Poisson sampler expects rate parameter
        return jax.random.poisson(key, rate, shape=(n,))[..., None]
