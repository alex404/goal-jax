"""Binomial distribution as an exponential family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from ...geometry import Analytic
from ...geometry.exponential_family.combinators import AnalyticProduct


@dataclass(frozen=True)
class Binomial(Analytic):
    """Binomial distribution with fixed number of trials over counts $x \\in \\{0, 1, \\ldots, n\\}$.

    $$p(x; \\theta) = \\binom{n}{x} p^x (1-p)^{n-x}$$

    where $p = \\sigma(\\theta)$ and $n$ is `n_trials`.

    As an exponential family:
        - Sufficient statistic: $s(x) = x$ (count)
        - Base measure: $\\log C(n, x) = \\log(n!) - \\log(x!) - \\log((n-x)!)$
        - Natural parameter: $\\theta = \\log(p/(1-p))$ (log odds)
        - Mean parameter: $\\eta = np = E[x]$
        - Log partition: $\\psi(\\theta) = n \\log(1 + \\exp(\\theta)) = n \\cdot \\text{softplus}(\\theta)$
        - Negative entropy: $n(p\\log p + (1-p)\\log(1-p))$
    """

    n_trials: int = 1
    """Number of trials (fixed)."""

    @property
    @override
    def dim(self) -> int:
        return 1

    @property
    @override
    def data_dim(self) -> int:
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return jnp.atleast_1d(x).astype(jnp.float32)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Binomial coefficient: $\\log C(n, x)$."""
        x = jnp.atleast_1d(x)[0]
        n = self.n_trials
        return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)

    @override
    def log_partition_function(self, params: Array) -> Array:
        return self.n_trials * jax.nn.softplus(params[0])

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Negative entropy: $n(p\\log p + (1-p)\\log(1-p))$ where $p = \\eta / n$."""
        n = self.n_trials
        mu = means[0]

        # Recover p from mean: mu = n * p
        p = jnp.clip(mu / n, 1e-10, 1 - 1e-10)

        # -H = n * (p*log(p) + (1-p)*log(1-p)) = n * Bernoulli_entropy(p)
        return n * (p * jnp.log(p) + (1 - p) * jnp.log(1 - p))

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        prob = jax.nn.sigmoid(params[0])
        samples = jax.random.binomial(key, self.n_trials, prob, shape=(n,))
        return samples.astype(jnp.float32).reshape(n, 1)

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize by shrinking sample means toward n/2, then converting to logits with noise."""
        avg_suff = self.average_sufficient_statistic(sample)
        # avg_suff[0] is mean count, convert to probability
        mean_p = avg_suff[0] / self.n_trials

        # Shrink toward 0.5 to handle exact 0s and 1s smoothly
        shrinkage = 0.01
        shrunk_p = (1 - shrinkage) * mean_p + shrinkage * 0.5

        # Convert to natural parameters (logits): theta = log(p/(1-p))
        shrunk_p = jnp.clip(shrunk_p, 1e-6, 1 - 1e-6)
        natural = jnp.atleast_1d(jnp.log(shrunk_p / (1 - shrunk_p)))

        noise = jax.random.normal(key, shape=(self.dim,)) * shape + location
        return natural + noise

    # Methods

    def to_prob(self, means: Array) -> Array:
        """Extract probability $p$ from mean parameters ($\\eta = np$)."""
        return means[0] / self.n_trials

    def from_prob(self, prob: float) -> Array:
        """Construct mean parameters from probability $p$ ($\\eta = np$)."""
        return jnp.array([self.n_trials * prob])


class Binomials(AnalyticProduct[Binomial]):
    """Product of $n$ independent Binomial distributions sharing the same `n_trials`, useful for modeling count data like grayscale pixel values."""

    def __init__(self, n_neurons: int, n_trials: int = 255):
        super().__init__(Binomial(n_trials), n_neurons)

    @property
    def n_neurons(self) -> int:
        """Number of Binomial units."""
        return self.n_reps

    @property
    def n_trials(self) -> int:
        """Number of trials for each unit."""
        return self.rep_man.n_trials
