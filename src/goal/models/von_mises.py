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

from ..geometry import Differentiable, Natural, Point


@dataclass(frozen=True)
class VonMises(Differentiable):
    """[previous docstring remains the same]"""

    @property
    def dim(self) -> int:
        return 2

    @property
    def data_dim(self) -> int:
        return 1

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        return jnp.array([jnp.cos(x), jnp.sin(x)]).ravel()

    def log_base_measure(self, x: Array) -> Array:
        return -jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        kappa = jnp.sqrt(jnp.sum(natural_params**2))
        # Explicitly cast i0e output to Array
        return jnp.log(i0e(kappa)) + kappa

    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        """Sample from von Mises distribution using direct transformation method.

        Based on Best & Fisher (1979) algorithm as implemented in numpy/numpyro.
        """
        mu, kappa = self.split_mean_concentration(p)
        kappa = jnp.maximum(kappa, 1e-5)

        # Split key for the two uniform samples needed
        key_u, key_v = jax.random.split(key)

        # Generate shape-(n,) uniform samples
        u = jax.random.uniform(key_u, (n,))
        v = jax.random.uniform(key_v, (n,))

        # Direct transformation of uniforms to von Mises
        z = jnp.cos(jnp.pi * u)
        r = (1.0 + jnp.sqrt(1.0 + 4.0 * kappa**2)) / (4.0 * kappa)
        w = (1.0 + r * z) / (r + z)

        samples = w * (2.0 * jnp.arccos(jnp.sqrt(w)) * (v - 0.5) + jnp.arcsin(z))

        # Shift by mean direction and ensure samples are in [-pi, pi]
        samples = samples + mu
        samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        return samples[..., None]

    def split_mean_concentration(self, p: Point[Natural, Self]) -> tuple[Array, Array]:
        theta = p.params
        kappa = jnp.sqrt(jnp.sum(theta**2))
        mu = jnp.arctan2(theta[1], theta[0])
        return mu, kappa

    def join_mean_concentration(
        self, mu0: float, kappa0: float
    ) -> Point[Natural, Self]:
        mu = jnp.atleast_1d(mu0)
        kappa = jnp.atleast_1d(kappa0)
        theta = kappa * jnp.concatenate([jnp.cos(mu), jnp.sin(mu)])
        return Point(theta)
