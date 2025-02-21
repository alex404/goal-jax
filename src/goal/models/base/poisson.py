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

#### Class Hierarchy

![Class Hierarchy](poisson.svg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    LocationShape,
    Mean,
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
    @override
    def dim(self) -> int:
        """Single rate parameter."""
        return 1

    @property
    @override
    def data_dim(self) -> int:
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        return self.mean_point(jnp.atleast_1d(x))

    @override
    def log_base_measure(self, x: Array) -> Array:
        k = jnp.asarray(x, dtype=jnp.float32)
        return -_log_factorial(k)

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        return jnp.squeeze(jnp.exp(params.array))

    @override
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        rate = means.array
        return jnp.squeeze(rate * (jnp.log(rate) - 1))

    @override
    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        mean_point = self.to_mean(params)
        rate = mean_point.array

        # JAX's Poisson sampler expects rate parameter
        return jax.random.poisson(key, rate, shape=(n,))[..., None]

    # Methods

    def statistical_mean(self, params: Point[Natural, Self]) -> Array:
        return self.to_mean(params).array.reshape([1])

    def statistical_covariance(self, params: Point[Natural, Self]) -> Array:
        return self.to_mean(params).array.reshape([1, 1])


@dataclass(frozen=True)
class CoMShape(ExponentialFamily):
    """Shape component of a CoMPoisson distribution.

    This represents the dispersion structure with sufficient statistic log(x!).
    """

    def __init__(self):
        super().__init__()

    @property
    @override
    def dim(self) -> int:
        return 1

    @property
    @override
    def data_dim(self) -> int:
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        return self.mean_point(jnp.atleast_1d(_log_factorial(x)))

    @override
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
    """

    # Fields

    window_size: int = 200

    # Override

    @property
    @override
    def fst_man(self) -> Poisson:
        return Poisson()

    @property
    @override
    def snd_man(self) -> CoMShape:
        return CoMShape()

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.asarray(0.0)

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Compute log partition function using fixed-width window strategy.

        Evaluates:
        $$\\psi(\\theta) = \\log\\sum_{j=0}^{\\infty} \\exp(\\theta_1 j + \\theta_2 \\log(j!))$$

        using a fixed number of terms centered on the mode.

        Args:
            params: Array of natural parameters $(\\theta_1, \\theta_2)$

        Returns:
            Value of log partition function $\\psi(\\theta)$
        """
        # Estimate mode and center window around it
        # Estimate mode
        mu, _ = self.split_mode_dispersion(params)

        # Create fixed window of indices
        base_indices = jnp.arange(self.window_size)

        # Shift window to be centered on mode (rounded down to integer)
        mode_shift = jnp.maximum(0, jnp.floor(mu - self.window_size / 2)).astype(
            jnp.int32
        )
        indices = base_indices + mode_shift

        def _compute_log_partition_terms(index: Array) -> Array:
            return self.dot(params, self.sufficient_statistic(index))

        # Compute terms and use log-sum-exp for numerical stability
        log_terms = jax.vmap(_compute_log_partition_terms)(indices)
        return jax.nn.logsumexp(log_terms)

    # TODO: Come up with a better scheme than rejection sampling
    @override
    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        """Generate random COM-Poisson samples using Algorithm 2 from Benson & Friel (2021)."""
        mu, nu = self.split_mode_dispersion(params)
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

                return key, y.reshape(()), accept

            init_val = (key, jnp.array(0), jnp.asarray(False).reshape(()))
            _, sample, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
            return sample

        keys = jax.random.split(key, n)
        samples = jax.vmap(sample_one)(keys)
        return samples[..., None]

    @override
    def check_natural_parameters(self, params: Point[Natural, Self]) -> Array:
        """Check if natural parameters are valid for COM-Poisson.

        For parameters $(\\theta_1, \\theta_2)$, the following conditions must hold:
        - $\\theta_1$ is finite, $\\theta_2 < 0$
        """
        finite = super().check_natural_parameters(params)
        theta2_valid = params[1] < 0
        return finite & theta2_valid

    @override
    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Point[Natural, Self]:
        """Initialize COM-Poisson parameters."""
        key_mu, key_nu = jax.random.split(key)

        # Ensure mu stays positive by using exp
        mu_init = jnp.exp(jax.random.normal(key_mu) * shape + location)

        # Keep nu in a reasonable range
        nu_init = 1.0 + jnp.abs(jax.random.normal(key_nu)) * shape

        return self.join_mode_dispersion(mu_init, nu_init)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize COM-Poisson parameters from sample.

        Estimates mode and shape parameters using method of moments based on sample mean and variance, with added noise for regularization.
        """
        # Compute sample statistics
        mean = jnp.mean(sample)
        var = jnp.var(sample)

        # Add noise for regularization
        noise = jax.random.normal(key, shape=(2,)) * shape + location
        mean = mean + noise[0]
        var = var + noise[1]

        a = var
        b = -(mean + 0.5)
        c = 0.5

        nu = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)

        mu = var * nu

        # Convert to natural parameters
        return self.join_mode_dispersion(mu, nu)

    # Methods

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

    def join_mode_dispersion(self, mu: Array, nu: Array) -> Point[Natural, Self]:
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
        return self.natural_point(jnp.array([theta1, theta2]).ravel())

    def approximate_mean_variance(
        self, natural_params: Point[Natural, Self]
    ) -> tuple[Array, Array]:
        """Compute approximate mean and variance of COM-Poisson distribution.

        Given mode $\\mu$ and shape $\\nu$ parameters, the approximations are:
        $E(X) \\approx \\mu + 1/(2\\nu) - 1/2$
        Var(X) ≈ $\\mu$/$\\nu$

        Args:
            natural_params: Natural parameters $(\\theta_1$, \\theta_2)$

        Returns:
            Tuple of (approximate mean, approximate variance)
        """
        mu, nu = self.split_mode_dispersion(natural_params)
        approx_mean = mu + 1 / (2 * nu) - 0.5
        approx_var = mu / nu
        return approx_mean, approx_var

    def numerical_mean_variance(
        self,
        natural_params: Point[Natural, Self],
    ) -> tuple[Array, Array]:
        """Compute mean and variance using numerical integration.

        Uses window-based approach centered on mode to compute:
        $$E[X] = \\sum_{x=0}^\\infty x p(x)$$
        $$E[X^2] = \\sum_{x=0}^\\infty x^2 p(x)$$
        $$\\text{Var}(X) = E[X^2] - E[X]^2$$

        Args:
            natural_params: Natural parameters $(\\theta_1, \\theta_2)$

        Returns:
            Tuple of (mean, variance)
        """
        # Get mode for window centering
        mu, _ = self.split_mode_dispersion(natural_params)

        # Create fixed window of indices
        mode_shift = jnp.maximum(0, jnp.floor(mu - self.window_size / 2)).astype(
            jnp.int32
        )
        indices = jnp.arange(self.window_size) + mode_shift

        # Compute log probabilities
        log_probs = jax.vmap(self.log_density, in_axes=(None, 0))(
            natural_params, indices
        )
        probs = jnp.exp(log_probs)

        # Compute first and second moments
        mean = jnp.sum(indices * probs)
        second_moment = jnp.sum(indices**2 * probs)

        # Compute variance
        variance = second_moment - mean**2

        return mean, variance

    def statistical_mean(self, params: Point[Natural, Self]) -> Array:
        mean, _ = self.numerical_mean_variance(params)
        return mean.reshape([1])

    def statistical_covariance(self, params: Point[Natural, Self]) -> Array:
        _, var = self.numerical_mean_variance(params)
        return var.reshape([1, 1])


def _log_factorial(k: Array) -> Array:
    return jax.lax.lgamma(k.astype(float) + 1)
