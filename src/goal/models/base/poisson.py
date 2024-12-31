"""
The Conway-Maxwell Poisson distribution is a generalization of the Poisson distribution
that can model both over- and under-dispersed count data. Its probability mass function is:

$$p(x; \\mu, \nu) = \\frac{\\mu^x}{(x!)^\\nu Z(\\mu, \\nu)}$$

where:

- $\\mu > 0$ is related to the mode of the distribution
- $\\nu > 0$ is the dispersion parameter (pseudo-precision)
- $Z(\\mu, \\nu)$ is the normalizing constant defined as:

$$Z(\\mu, \\nu) = \\sum_{j=0}^{\\infty} \\frac{\\mu^j}{(j!)^\\nu}$$

As an exponential family, it can be written with:

- Natural parameters: $\\theta_1 = \\nu\\log(\\mu)$, $\\theta_2 = -\\nu$
- Base measure: $\\mu(x) = 0$
- Sufficient statistics: $s(x) = (x, \\log(x!))$

Natural to mode-shape:
$(\\theta_1, \\theta_2) \\mapsto (\\exp(-\\theta_1/\\theta_2), -\\theta_2)$

Mode-shape to natural:
$(\\mu, \\nu) \\mapsto (\\nu\\log(\\mu), -\\nu)$

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

from ...geometry import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    LocationShape,
    Natural,
    Point,
)


@dataclass(frozen=True)
class Poisson(Analytic):
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
        return jnp.atleast_1d(x)

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

    # Shape/data initialization based on approximate mean and variance


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
    """

    window_size: int = 200

    def __init__(self, window_size: int = 200):
        super().__init__(Poisson(), CoMShape())
        object.__setattr__(self, "window_size", window_size)

    def split_mode_dispersion(
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

    def join_mode_dispersion(self, mu: float, nu: float) -> Point[Natural, Self]:
        """Convert from mode-shape parameters to natural parameters.

        The COM-Poisson distribution can be parameterized by either mode-shape parameters $(\\mu, \\nu)$ or natural parameters $(\\theta_1, \\theta_2)$. The conversion
        is given by:

        - $\\theta_1 = \\nu\\log(\\mu)$
        - $\\theta_2 = -\\nu$

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
        mu, _ = self.split_mode_dispersion(Point(natural_params))

        # Create fixed window of indices
        base_indices = jnp.arange(self.window_size)

        # Shift window to be centered on mode (rounded down to integer)
        mode_shift = jnp.maximum(0, jnp.floor(mu - self.window_size / 2)).astype(
            jnp.int32
        )
        indices = base_indices + mode_shift

        def _compute_log_partition_terms(natural_params: Array, index: Array) -> Array:
            return self.dot(self.sufficient_statistic(index), Point(natural_params))

        # Compute terms and use log-sum-exp for numerical stability
        log_terms = jax.vmap(_compute_log_partition_terms, in_axes=(None, 0))(
            natural_params, indices
        )
        return jax.nn.logsumexp(log_terms)

    # TODO: Come up with a better scheme than rejection sampling
    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        """Generate random COM-Poisson samples using Algorithm 2 from Benson & Friel (2021)."""
        mu, nu = self.split_mode_dispersion(p)
        mode = jnp.floor(mu)

        # Envelope terms for both Poisson and Geometric cases

        # Underdispersed case (nu >= 1): Poisson envelope
        log_pois_scale = (nu - 1) * (mode * jnp.log(mu) - _log_factorial(mode))

        # Overdispersed case (nu < 1): Geometric envelope
        p_geo = (2 * nu) / (2 * nu * mu + 1 + nu)
        ratio = jnp.floor(mu / ((1 - p_geo) ** (1 / nu)))
        log_geo_scale = (
            -jnp.log(p_geo)
            + nu * ratio * jnp.log(mu)
            - (ratio * jnp.log(1 - p_geo) + nu * _log_factorial(ratio))
        )

        def sample_one(key: Array) -> Array:
            def cond_fn(val: tuple[Array, Array, Array]) -> Array:
                _, _, accept = val
                return jnp.logical_not(accept)

            def body_fn(val: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
                key, y, _ = val
                key, key_prop, key_u = jax.random.split(key, 3)

                # Underdispersed case (nu >= 1): Poisson envelope
                pois_y = jax.random.poisson(key_prop, mu)
                log_alpha_pois = nu * (
                    pois_y * jnp.log(mu) - _log_factorial(pois_y)
                ) - (log_pois_scale + pois_y * jnp.log(mu) - _log_factorial(pois_y))

                # Overdispersed case (nu >= 1): Poisson envelope
                u0 = jax.random.uniform(key_prop)
                geo_y = jnp.floor(jnp.log(u0) / jnp.log(1 - p_geo))
                log_alpha_geo = nu * (geo_y * jnp.log(mu) - _log_factorial(geo_y)) - (
                    log_geo_scale + geo_y * jnp.log(1 - p_geo) + jnp.log(p_geo)
                )

                # Select proposal and ratio based on nu
                y = jnp.where(nu >= 1, pois_y, geo_y)
                log_alpha = jnp.where(nu >= 1, log_alpha_pois, log_alpha_geo)

                # Accept/reject step
                u = jax.random.uniform(key_u)
                accept = jnp.asarray(jnp.log(u) <= log_alpha).reshape(())

                return key, y, accept

            init_val = (key, jnp.array(0), jnp.asarray(False).reshape(()))
            _, sample, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
            return sample

        keys = jax.random.split(key, n)
        samples = jax.vmap(sample_one)(keys)
        return samples[..., None]


def _log_factorial(n: Array) -> Array:
    return jnp.atleast_1d(jax.lax.lgamma(n.astype(float) + 1))
