"""
The Conway-Maxwell Poisson distribution is a generalization of the Poisson distribution
that can model both over- and under-dispersed count data. Its probability mass function is:

$$p(x; \\mu, \nu) = \\frac{\\mu^x}{(x!)^\\nu Z(\\mu, \\nu)}$$

where:
- $\\mu > 0$ is related to the mode of the distribution
- $\\nu > 0$ is the dispersion parameter (pseudo-precision)
- $Z(\\mu, \\nu)$ is the normalizing constant defined as:

$$Z(\\mu, \\nu) = \\sum_{j=0}^{\\infty} \frac{\\mu^j}{(j!)^\\nu}$$

As an exponential family, it can be written with:
- Natural parameters: $\theta_1 = \\nu\\log(\\mu)$, $\theta_2 = -\\nu$
- Base measure: $\\mu(x) = 0$
- Sufficient statistics: $s(x) = (x, \\log(x!))$

Natural to mode-shape:
$$(\\theta_1, \\theta_2) \\mapsto (\\exp(-\\theta_1/\\theta_2), -\\theta_2)$$

Mode-shape to natural:
$$(\\mu, \\nu) \\mapsto (\\nu\\log(\\mu), -\\nu)$$

Key numerical considerations:

1. Series evaluation strategy balancing accuracy and performance
2. Mode-directed evaluation to handle both under- and over-dispersed cases
3. JAX-optimized computation patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from ..geometry import Differentiable, ExponentialFamily, LocationShape, Natural, Point
from .univariate import Poisson


@dataclass(frozen=True)
class CoMShape(ExponentialFamily):
    """Shape component of a CoMPoisson distribution.

    This represents the dispersion structure with sufficient statistic log(x!).
    """

    def __init__(self):
        super().__init__()

    @property
    def dim(self) -> int:
        return 1

    @property
    def data_dim(self) -> int:
        return 1

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        return _log_factorial(x)

    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)


@dataclass(frozen=True)
class CoMPoisson(LocationShape[Poisson, CoMShape], Differentiable):
    """Conway-Maxwell Poisson distribution implementation optimized for JAX.

    This implementation uses fixed-width window numerical integration centered on the
    approximate mode of the distribution. This approach provides:

    1. JAX-optimized performance through static computation graphs
    2. Numerically stable computation in log space
    3. Accurate evaluation for both under- and over-dispersed cases

    Args:
        window_size: Fixed number of terms to evaluate in series expansions (default: 200)
        max_proposals: Maximum number of proposals for rejection sampling (default: 10)
    """

    window_size: int = 200
    max_proposals: int = 10

    def __init__(self, window_size: int = 200):
        super().__init__(Poisson(), CoMShape())
        object.__setattr__(self, "window_size", window_size)

    def split_mode_shape(
        self, natural_params: Point[Natural, Self]
    ) -> tuple[Array, Array]:
        """Convert from natural parameters to mode-shape parameters.

        The COM-Poisson distribution can be parameterized by either natural parameters $(\\theta_1, \\theta_2)$ or by mode-shape parameters $(\\mu, \\nu)$. The conversion
        is given by:

        $$\\nu = -\\theta_2$$
        $$\\mu = \\exp(-\\theta_1/\\theta_2)$$

        Args:
            natural_params: Natural parameters $(\\theta_1, \\theta_2)$

        Returns:
            Tuple of mode parameter $\\mu$ and shape parameter $\\nu$
        """
        theta1, theta2 = natural_params[0], natural_params[1]
        nu = -theta2
        mu = jnp.exp(-theta1 / theta2)
        return mu, nu

    def join_mode_shape(self, mu: float, nu: float) -> Point[Natural, Self]:
        """Convert from mode-shape parameters to natural parameters.

        The COM-Poisson distribution can be parameterized by either mode-shape parameters $(\\mu, \\nu)$ or natural parameters $(\\theta_1, \\theta_2)$. The conversion
        is given by:

        $$\\theta_1 = \\nu\\log(\\mu)$$
        $$\\theta_2 = -\\nu$$

        Args:
            mu: Mode parameter $\\mu > 0$
            nu: Shape parameter $\\nu > 0$

        Returns:
            Natural parameters $(\\theta_1, \\theta_2)$ as a Point
        """
        theta1 = nu * jnp.log(mu)
        theta2 = -nu
        return Point(jnp.array([theta1, theta2]))

    def log_base_measure(self, x: Array) -> Array:
        return jnp.asarray(0.0)

    def _compute_log_partition_terms(
        self, natural_params: Array, indices: Array
    ) -> Array:
        """Compute terms in the log partition function for given indices.

        For natural parameters $(\\theta_1, \\theta_2)$, computes array of:
        $$\\theta_1 j + \\theta_2 \\log(j!)$$

        Args:
            natural_params: Array of natural parameters $(\\theta_1, \\theta_2)$
            indices: Array of indices $j$ to evaluate terms at

        Returns:
            Array of log terms in the partition function
        """
        theta1, theta2 = natural_params[0], natural_params[1]
        log_factorial = _log_factorial(indices + 1)
        return theta1 * indices + theta2 * log_factorial

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        """Compute log partition function using fixed-width window strategy.

        Evaluates:
        $$\\psi(\\theta) = \\log\\sum_{j=0}^{\\infty} \\exp(\\theta_1 j + \\theta_2 \\log(j!))$$

        using a fixed number of terms centered on the mode.

        Args:
            natural_params: Array of natural parameters $(\\theta_1, \\theta_2)$

        Returns:
            Value of log partition function $\\psi(\\theta)$
        """
        # Estimate mode and center window around it
        # Estimate mode
        mu, _ = self.split_mode_shape(Point(natural_params))

        # Create fixed window of indices
        base_indices = jnp.arange(self.window_size)

        # Shift window to be centered on mode (rounded down to integer)
        mode_shift = jnp.maximum(0, jnp.floor(mu - self.window_size / 2)).astype(
            jnp.int32
        )
        indices = base_indices + mode_shift

        # Compute terms and use log-sum-exp for numerical stability
        log_terms = self._compute_log_partition_terms(natural_params, indices)
        return jax.nn.logsumexp(log_terms)

    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        """Generate random samples from the distribution.

        Args:
            key: PRNGKey that will be split into (n, batch_shape, 2) keys
            p: Parameters in natural coordinates
            n: Number of samples per parameter set

        Returns:
            Array of shape (n, *data_dims) containing samples
        """
        mu, nu = self.split_mode_shape(p)

        # Split into n sets of batch_size pairs of keys
        keys = jax.random.split(key, n * 2)
        proposal_keys = keys[::2].reshape(n, *mu.shape, -1)
        accept_keys = keys[1::2].reshape(n, *mu.shape, -1)

        def sample_one(pkey: Array, akey: Array, mu: Array, nu: Array) -> Array:
            is_poisson = nu >= 1
            return jnp.where(
                is_poisson,
                _poisson_envelope_sample_one(pkey, akey, mu, self.max_proposals),
                _geometric_envelope_sample_one(pkey, akey, mu, nu, self.max_proposals),
            )

        return jax.vmap(sample_one, in_axes=(0, 0, None, None))(
            proposal_keys, accept_keys, mu, nu
        ).reshape(n, 1)


def _log_factorial(n: Array) -> Array:
    return jnp.atleast_1d(jax.lax.lgamma(n.astype(float) + 1))


def _safe_log(x: Array) -> Array:
    """Safe log that handles x = 0."""
    return jnp.log(jnp.maximum(x, 1e-10))


def _poisson_envelope_sample_one(
    proposal_key: Array,
    accept_key: Array,
    mu: Array,
    max_proposals: int,
) -> Array:
    """Sample single value from Poisson envelope with fixed number of proposals."""
    proposals = jax.random.poisson(proposal_key, mu, shape=(max_proposals,))
    log_mu = _safe_log(mu)
    log_fact = jax.vmap(_log_factorial)(proposals)
    ratios = proposals * log_mu - log_fact

    uniforms = jax.random.uniform(accept_key, shape=(max_proposals,))
    accepted = uniforms < ratios

    first_accept = jnp.argmax(accepted)
    return proposals[jnp.minimum(first_accept, max_proposals - 1)]


def _geometric_envelope_sample_one(
    proposal_key: Array,
    accept_key: Array,
    mu: Array,
    nu: Array,
    max_proposals: int,
) -> Array:
    """Sample single value from geometric envelope with fixed number of proposals."""
    p = 2 * nu / (2 * mu * nu + 1 + nu)

    uniforms = jax.random.uniform(proposal_key, shape=(max_proposals,))
    proposals = jnp.floor(_safe_log(uniforms) / _safe_log(1 - p))

    log_mu = _safe_log(mu)
    log_fact = jax.vmap(_log_factorial)(proposals)
    ratios = nu * (proposals * log_mu - log_fact)
    bound = nu * (proposals * _safe_log(1 - p))

    accept_uniforms = jax.random.uniform(accept_key, shape=(max_proposals,))
    accepted = accept_uniforms < jnp.exp(ratios - bound)

    first_accept = jnp.argmax(accepted)
    return proposals[jnp.minimum(first_accept, max_proposals - 1)]

    # def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
    #     """Generate random samples using rejection sampling as in Benson & Friel (2021)."""
    #     mu, nu = self.split_mode_shape(p)
    #
    #     def sample_one(key: Array) -> Array:
    #         def cond_fn(val: tuple[Array, Array, Array]) -> Array:
    #             _, _, accept = val
    #             return jnp.logical_not(accept)  # accept is scalar shape ()
    #
    #         def body_fn(val: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    #             key, y, _ = val
    #             key, key_prop, key_u = jax.random.split(key, 3)
    #
    #             # Poisson case
    #             poisson_y = jax.random.poisson(key_prop, mu)
    #             poisson_log_ratio = -(nu - 1) * jnp.log(mu) + (nu - 1) * _log_factorial(
    #                 poisson_y
    #             )
    #
    #             # Geometric case
    #             p = 2 * nu / (2 * mu * nu + 1 + nu)
    #             u0 = jax.random.uniform(key_prop)
    #             geom_y = jnp.floor(jnp.log(u0) / jnp.log(1 - p))
    #             geom_log_ratio = nu * (
    #                 geom_y * jnp.log(mu) - _log_factorial(geom_y)
    #             ) - (geom_y * jnp.log(1 - p))
    #
    #             # Select based on nu
    #             y = jnp.where(nu >= 1, poisson_y, geom_y)
    #             log_ratio = jnp.where(nu >= 1, poisson_log_ratio, geom_log_ratio)
    #
    #             # Accept/reject ensures scalar result of shape ()
    #             u = jax.random.uniform(key_u)
    #             accept = jnp.asarray(jnp.log(u) <= log_ratio).reshape(
    #                 ()
    #             )  # Force scalar
    #
    #             return key, y, accept
    #
    #         init_val = (
    #             key,
    #             jnp.array(0),
    #             jnp.asarray(False).reshape(()),
    #         )  # Force scalar
    #         _, sample, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
    #         return sample
    #
    #     keys = jax.random.split(key, n)
    #     samples = jax.vmap(sample_one)(keys)
    #     return samples[..., None]
    #
    #
    #
