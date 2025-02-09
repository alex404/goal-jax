"""Core exponential families. In particular provides

- `Categorical`: Family of discrete probability distributions over $n$ states.
- `Poisson`: Family of Poisson distribution over counts $k \\in \\mathbb{N}$.


#### Class Hierarchy

![Class Hierarchy](categorical.svg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import Analytic, Mean, Natural, Point


@dataclass(frozen=True)
class Categorical(Analytic):
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
    @override
    def dim(self) -> int:
        """Dimension $d$ is (`n_categories` - 1) due to the sum-to-one constraint."""
        return self.n_categories - 1

    @property
    @override
    def data_dim(self) -> int:
        """Dimension of the data space."""
        return 1

    # Categorical methods

    def from_probs(self, probs: Array) -> Point[Mean, Self]:
        """Construct the mean parameters from the complete probabilities, dropping the first element."""
        return self.mean_point(probs[1:])

    def to_probs(self, p: Point[Mean, Self]) -> Array:
        """Return the probabilities of all labels."""
        probs = p.array
        prob0 = 1 - jnp.sum(probs)
        return jnp.concatenate([jnp.array([prob0]), probs])

    # Exponential family methods

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        return self.mean_point(jax.nn.one_hot(x - 1, self.n_categories - 1).reshape(-1))

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        array = jnp.concatenate([jnp.array([0.0]), params.array])
        max_val = jnp.max(array)
        return max_val + jax.nn.logsumexp(array - max_val)

    @override
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        probs = self.to_probs(means)
        return jnp.sum(probs * jnp.log(probs))

    @override
    def sample(
        self,
        key: Array,
        params: Point[Natural, Self],
        n: int = 1,
    ) -> Array:
        mean_point = self.to_mean(params)
        probs = self.to_probs(mean_point)

        key = jnp.asarray(key)
        # Use Gumbel-Max trick: argmax(log(p) + Gumbel(0,1)) ~ Categorical(p)
        g = jax.random.gumbel(key, shape=(n, self.n_categories))
        return jnp.argmax(jnp.log(probs) + g, axis=-1)[..., None]
