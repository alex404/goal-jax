"""Binomial distribution as exponential family.

This module provides:
- `Binomial`: Distribution over counts (0, 1, ..., n_trials) with fixed n_trials
- `Binomials`: Product of n independent Binomial distributions (same n_trials)
"""

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
    """Binomial distribution with fixed number of trials.

    The distribution over counts x in {0, 1, ..., n_trials} is:

    .. math::

        p(x; \\theta) = \\binom{n}{x} p^x (1-p)^{n-x}

    where p = sigmoid(theta) and n = n_trials.

    As an exponential family:
        - Sufficient statistic: s(x) = x (count)
        - Base measure: log C(n, x) = log(n!) - log(x!) - log((n-x)!)
        - Natural parameter: theta = log(p/(1-p)) (log odds)
        - Mean parameter: eta = n * p = E[x]
        - Log partition: psi(theta) = n * log(1 + exp(theta)) = n * softplus(theta)
        - Negative entropy: computed via exact sum over support

    Note: This is the same structure as Bernoulli but scaled by n_trials,
    with a non-zero base measure (binomial coefficients).

    Attributes:
        n_trials: Number of trials (n parameter of binomial)
    """

    n_trials: int = 1
    """Number of trials (fixed)."""

    @property
    @override
    def dim(self) -> int:
        """Parameter dimension is 1."""
        return 1

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension is 1 (single count value)."""
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic s(x) = x (the count)."""
        return jnp.atleast_1d(x).astype(jnp.float32)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure is binomial coefficient: log C(n, x)."""
        x = jnp.atleast_1d(x)[0]
        n = self.n_trials
        # log C(n, x) = log(n!) - log(x!) - log((n-x)!)
        return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Log partition function: n * log(1 + exp(theta)) = n * softplus(theta)."""
        return self.n_trials * jax.nn.softplus(params[0])

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Negative entropy in exponential family convention.

        For exponential families: -H = <mean, natural> - log_partition
        This excludes the base measure contribution for consistency with
        the relative_entropy computation.

        For Binomial with mean mu = n*p and natural theta = logit(p):
            -H = mu * theta - n * softplus(theta)
               = n * (p * log(p) + (1-p) * log(1-p))
        """
        n = self.n_trials
        mu = means[0]

        # Recover p from mean: mu = n * p
        p = jnp.clip(mu / n, 1e-10, 1 - 1e-10)

        # -H = n * (p*log(p) + (1-p)*log(1-p)) = n * Bernoulli_entropy(p)
        return n * (p * jnp.log(p) + (1 - p) * jnp.log(1 - p))

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from Binomial distribution.

        Args:
            key: JAX random key
            params: Natural parameters (log odds)
            n: Number of samples

        Returns:
            Array of shape (n, 1) with count values in {0, ..., n_trials}
        """
        prob = jax.nn.sigmoid(params[0])
        # jax.random.binomial expects (key, n_trials, p, shape)
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
        """Initialize Binomial parameters from sample data.

        Shrinks sample means toward n/2 (maximum entropy point) to handle
        boundary cases, converts to natural parameters (logits), then adds noise.

        Args:
            key: Random key
            sample: Sample data (count values)
            location: Mean of noise distribution
            shape: Std dev of noise distribution

        Returns:
            Natural parameters (log-odds).
        """
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

    # Convenience methods

    def to_prob(self, means: Array) -> Array:
        """Extract probability p from mean parameters (mean = n * p)."""
        return means[0] / self.n_trials

    def from_prob(self, prob: float) -> Array:
        """Construct mean parameters from probability p (mean = n * p)."""
        return jnp.array([self.n_trials * prob])


class Binomials(AnalyticProduct[Binomial]):
    """Product of n independent Binomial distributions.

    All units share the same n_trials parameter. This is useful for
    modeling count data like grayscale pixel values (0-255).

    Attributes:
        n_neurons: Number of independent Binomial units
        n_trials: Number of trials for each unit
    """

    def __init__(self, n_neurons: int, n_trials: int = 255):
        """Create a product of n independent Binomials.

        Args:
            n_neurons: Number of units
            n_trials: Number of trials for each unit (default 255 for grayscale)
        """
        super().__init__(Binomial(n_trials), n_neurons)

    @property
    def n_neurons(self) -> int:
        """Number of Binomial units."""
        return self.n_reps

    @property
    def n_trials(self) -> int:
        """Number of trials for each unit."""
        return self.rep_man.n_trials
