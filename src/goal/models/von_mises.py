"""Von Mises distribution over the circle.

The von Mises distribution is a continuous probability distribution on the circle, analogous to the normal distribution on the line. The probability density function is:

$$p(x; \\mu, \\kappa) = \\frac{1}{2\\pi I_0(\\kappa)}\\exp(\\kappa \\cos(x - \\mu))$$

where:
- $\\mu$ is the mean direction
- $\\kappa$ is the concentration parameter
- $I_0(\\kappa)$ is the modified Bessel function of the first kind of order 0

As an exponential family, it can be written with:
- Sufficient statistic: $\\mathbf{s}(x) = (\\cos(x), \\sin(x))$
- Base measure: $\\mu(x) = -\\log(2\\pi)$
- Natural parameters: $\\theta = \\kappa(\\cos(\\mu), \\sin(\\mu))$
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import i0e  # type: ignore

from ..geometry import Forward, Natural, Point


@dataclass(frozen=True)
class VonMises(Forward):
    """[previous docstring remains the same]"""

    @property
    def dim(self) -> int:
        return 2

    @property
    def data_dim(self) -> int:
        return 1

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        x = jnp.asarray(x).reshape(-1)
        return jnp.array([jnp.cos(x), jnp.sin(x)])

    def log_base_measure(self, x: Array) -> Array:
        return -jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        kappa = jnp.sqrt(jnp.sum(natural_params**2))
        # Explicitly cast i0e output to Array
        return jnp.log(i0e(kappa)) + kappa

    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        mu, kappa = self.to_mean_concentration(p)
        kappa = jnp.maximum(kappa, 1e-5)

        tau = 1 + jnp.sqrt(1 + 4 * kappa**2)
        rho = (tau - jnp.sqrt(2 * tau)) / (2 * kappa)
        r = (1 + rho**2) / (2 * rho)

        def sample_one(key: Array) -> Array:
            key1, key2, key3 = jax.random.split(key, 3)
            u1 = jax.random.uniform(key1)
            u2 = jax.random.uniform(key2)
            u3 = jax.random.uniform(key3)

            z = jnp.cos(jnp.pi * u1)
            f = (1 + r * z) / (r + z)
            c = kappa * (r - f)

            accept = jnp.log(c / u2) + 1 - c >= 0
            angle = jnp.where(u3 < 0.5, -jnp.arccos(f), jnp.arccos(f))
            # Add explicit typing for the recursive call
            result: Array = jnp.where(
                accept, angle + mu, sample_one(jax.random.split(key, 1)[0])
            )
            return result

        keys = jax.random.split(key, n)
        samples = jax.vmap(sample_one)(keys)
        return samples[..., None]  # Add explicit dimension at the end

    def to_mean_concentration(self, p: Point[Natural, Self]) -> tuple[Array, Array]:
        theta = p.params
        kappa = jnp.sqrt(jnp.sum(theta**2))
        mu = jnp.arctan2(theta[1], theta[0])
        return mu, kappa

    def from_mean_concentration(self, mu: Array, kappa: Array) -> Point[Natural, Self]:
        theta = kappa * jnp.array([jnp.cos(mu), jnp.sin(mu)])
        return Point(theta)
