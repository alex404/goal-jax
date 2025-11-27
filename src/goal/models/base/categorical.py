"""The categorical distribution (probability simplex) models distributions over finite sets of elements, and has sufficient parameters to model any such distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import Analytic


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

    def to_probs(self, p: Array) -> Array:
        """Return the probabilities of all labels."""
        prob0 = 1 - jnp.sum(p)
        return jnp.concatenate([jnp.array([prob0]), p])

    # Overrides

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return jax.nn.one_hot(x - 1, self.n_categories - 1).reshape(-1)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, natural_params: Array) -> Array:
        array = jnp.concatenate([jnp.array([0.0]), natural_params])
        max_val = jnp.max(array)
        return max_val + jax.nn.logsumexp(array - max_val)

    @override
    def negative_entropy(self, mean_params: Array) -> Array:
        probs = self.to_probs(mean_params)
        return jnp.sum(probs * jnp.log(probs))

    @override
    def sample(
        self,
        key: Array,
        natural_params: Array,
        n: int = 1,
    ) -> Array:
        mean_params = self.to_mean(natural_params)
        probs = self.to_probs(mean_params)

        key = jnp.asarray(key)
        # Use Gumbel-Max trick: argmax(log(p) + Gumbel(0,1)) ~ Categorical(p)
        g = jax.random.gumbel(key, shape=(n, self.n_categories))
        return jnp.argmax(jnp.log(probs) + g, axis=-1)[..., None]
