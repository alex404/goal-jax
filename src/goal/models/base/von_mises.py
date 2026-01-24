"""Von Mises distribution over the circle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import i0e

from ...geometry import Differentiable
from ...geometry.exponential_family.combinators import DifferentiableProduct


@dataclass(frozen=True)
class VonMises(Differentiable):
    """The von Mises distribution is a continuous probability distribution on the circle, analogous to the normal distribution on the line. The probability density function is:

    $$p(x; \\mu, \\kappa) = \\frac{1}{2\\pi I_0(\\kappa)}\\exp(\\kappa \\cos(x - \\mu))$$

    where:

    - $\\mu$ is the mean direction
    - $\\kappa$ is the concentration parameter
    - $I_0(\\kappa)$ is the modified Bessel function of the first kind of order 0

    As an exponential family:

    - Sufficient statistic: $\\mathbf{s}(x) = (\\cos(x), \\sin(x))$
    - Base measure: $\\mu(x) = -\\log(2\\pi)$
    - Natural parameters: $\\theta = \\kappa(\\cos(\\mu), \\sin(\\mu))$
    """

    # Methods

    def split_mean_concentration(self, p: Array) -> tuple[Array, Array]:
        """Split the natural parameters into mean and concentration parameters.

        Args:
            p: Natural parameters array

        Returns:
            Tuple of (mean_angle, concentration)
        """
        theta = p
        kappa = jnp.sqrt(jnp.sum(theta**2))
        mu = jnp.arctan2(theta[1], theta[0])
        return mu, kappa

    def join_mean_concentration(self, mu0: float, kappa0: float) -> Array:
        """Join the mean and concentration parameters into natural parameters.

        Args:
            mu0: Mean angle
            kappa0: Concentration parameter

        Returns:
            Natural parameters array
        """
        mu = jnp.atleast_1d(mu0)
        kappa = jnp.atleast_1d(kappa0)
        return kappa * jnp.concatenate([jnp.cos(mu), jnp.sin(mu)])

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return 2

    @property
    @override
    def data_dim(self) -> int:
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Compute sufficient statistics: (cos(x), sin(x)).

        Args:
            x: Data point

        Returns:
            Sufficient statistics array
        """
        return jnp.array([jnp.cos(x), jnp.sin(x)]).ravel()

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Log base measure: -log(2π).

        Args:
            x: Data point

        Returns:
            Log base measure (scalar)
        """
        return -jnp.log(2 * jnp.pi)

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Compute log partition function.

        Args:
            params: Natural parameters

        Returns:
            Log partition function value (scalar)
        """
        kappa = jnp.sqrt(jnp.sum(params**2))
        # Explicitly cast i0e output to Array
        return jnp.log(i0e(kappa)) + kappa

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Generate n samples from the Von Mises distribution.

        Uses batched rejection sampling with wrapped Cauchy proposal,
        adapted from NumPyro's implementation (Devroye, 1986).

        For very small concentration (kappa < 0.01), samples uniformly
        on the circle since the distribution is nearly uniform.

        Args:
            key: JAX random key
            params: Natural parameters
            n: Number of samples

        Returns:
            Array of n samples with shape (n, 1)
        """
        mu, kappa = self.split_mean_concentration(params)

        # For very small kappa, the distribution is nearly uniform
        # Use uniform sampling to avoid numerical issues
        kappa_threshold = 0.01
        use_uniform = kappa < kappa_threshold

        # Clamp kappa for rejection sampling (used when kappa >= threshold)
        kappa_clamped = jnp.maximum(kappa, kappa_threshold)

        # Compute shape parameter s with numerical stability
        # For small kappa, use approximate formula; for large kappa, use exact
        s_cutoff = 1.2e-4  # Cutoff for float64
        r = 1.0 + jnp.sqrt(1.0 + 4.0 * kappa_clamped**2)
        rho = (r - jnp.sqrt(2.0 * r)) / (2.0 * kappa_clamped)
        s_exact = (1.0 + rho**2) / (2.0 * rho)
        s_approximate = 1.0 / kappa_clamped
        s = jnp.where(kappa_clamped > s_cutoff, s_exact, s_approximate)

        # Broadcast s to sample shape
        shape = (n,)
        s = jnp.broadcast_to(s, shape)
        kappa_broadcast = jnp.broadcast_to(kappa_clamped, shape)

        def cond_fn(val: tuple[Array, Array, Array, Array, Array]) -> Array:
            """Check if all samples done or reached max iterations."""
            i, _, done, _, _ = val
            return jnp.logical_and(i < 100, jnp.logical_not(jnp.all(done)))

        def body_fn(
            val: tuple[Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array, Array]:
            i, key, done, u, w = val
            key, key_u, key_v = jax.random.split(key, 3)

            # Sample uniform in [-1, 1] and [0, 1]
            u_new = jax.random.uniform(key_u, shape=shape, minval=-1.0, maxval=1.0)
            v = jax.random.uniform(key_v, shape=shape)

            # Wrapped Cauchy proposal
            z = jnp.cos(jnp.pi * u_new)
            w_new = (1.0 + s * z) / (s + z)

            # Acceptance criterion
            y = kappa_broadcast * (s - w_new)
            # Add small epsilon to avoid log(0) or division issues
            y_safe = jnp.maximum(y, 1e-10)
            v_safe = jnp.maximum(v, 1e-10)
            accept = jnp.logical_or(
                y_safe * (2.0 - y_safe) >= v_safe,
                jnp.log(y_safe / v_safe) + 1.0 >= y_safe,
            )

            # Update only where not already done
            u = jnp.where(done, u, u_new)
            w = jnp.where(done, w, w_new)
            done = jnp.logical_or(done, accept)

            return i + 1, key, done, u, w

        # Initialize rejection sampling loop
        init_done = jnp.zeros(shape, dtype=bool)
        init_u = jnp.zeros(shape)
        init_w = jnp.zeros(shape)
        init_val = (jnp.array(0), key, init_done, init_u, init_w)

        _, _, _, u, w = jax.lax.while_loop(cond_fn, body_fn, init_val)

        # Convert to angle and shift by mean (rejection sampling result)
        rejection_samples = jnp.sign(u) * jnp.arccos(jnp.clip(w, -1.0, 1.0)) + mu

        # Uniform samples on the circle
        key, uniform_key = jax.random.split(key)
        uniform_samples = jax.random.uniform(
            uniform_key, shape=shape, minval=-jnp.pi, maxval=jnp.pi
        )

        # Use uniform samples when kappa is very small
        samples = jnp.where(use_uniform, uniform_samples, rejection_samples)
        return samples[..., None]


class VonMisesProduct(DifferentiableProduct[VonMises]):
    """Product of n independent Von Mises distributions.

    Useful for modeling multiple circular/angular latent variables.
    Each component has natural parameters (κ·cos(μ), κ·sin(μ)) where
    μ is the mean direction and κ is the concentration.

    The sufficient statistic for n components is a 2n-dimensional vector:
        [cos(θ_1), sin(θ_1), cos(θ_2), sin(θ_2), ..., cos(θ_n), sin(θ_n)]

    Attributes:
        n_components: Number of independent Von Mises variables
    """

    def __init__(self, n_components: int):
        """Create a product of n independent Von Mises distributions.

        Args:
            n_components: Number of Von Mises variables
        """
        super().__init__(VonMises(), n_components)

    @property
    def n_components(self) -> int:
        """Number of Von Mises components."""
        return self.n_reps
