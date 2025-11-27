"""Von Mises distribution over the circle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import i0e

from ...geometry import Differentiable


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
    def log_partition_function(self, natural_params: Array) -> Array:
        """Compute log partition function.

        Args:
            params: Natural parameters

        Returns:
            Log partition function value (scalar)
        """
        kappa = jnp.sqrt(jnp.sum(natural_params**2))
        # Explicitly cast i0e output to Array
        return jnp.log(i0e(kappa)) + kappa

    # TODO: Right now this is based on rejection sampling, but we should switch to a more efficient method that's more suitable for JAX
    @override
    def sample(self, key: Array, natural_params: Array, n: int = 1) -> Array:
        """Generate n samples from the Von Mises distribution.

        Args:
            key: JAX random key
            params: Natural parameters
            n: Number of samples

        Returns:
            Array of n samples
        """
        mu, kappa = self.split_mean_concentration(natural_params)
        kappa = jnp.maximum(kappa, 1e-5)

        tau = 1 + jnp.sqrt(1 + 4 * kappa**2)
        rho = (tau - jnp.sqrt(2 * tau)) / (2 * kappa)
        r = (1 + rho**2) / (2 * rho)

        def sample_one(key: Array) -> Array:
            def cond_fn(val: tuple[Array, Array, Array]) -> Array:
                _, _, accept = val
                return jnp.logical_not(accept)

            def body_fn(val: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
                key, _, _ = val
                key, key1, key2, key3 = jax.random.split(key, 4)

                u1 = jax.random.uniform(key1)
                u2 = jax.random.uniform(key2)
                u3 = jax.random.uniform(key3)

                z = jnp.cos(jnp.pi * u1)
                f = (1 + r * z) / (r + z)
                c = kappa * (r - f)

                accept = jnp.log(c / u2) + 1 - c >= 0
                new_angle = jnp.where(u3 < 0.5, -jnp.arccos(f), jnp.arccos(f))
                return key, new_angle, accept

            init_val = (key, jnp.array(0.0), jnp.array(False))
            _, angle, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
            return angle + mu

        keys = jax.random.split(key, n)
        samples = jax.vmap(sample_one)(keys)
        return samples[..., None]
