"""Dirichlet distribution as an exponential family over the probability simplex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import Differentiable


@dataclass(frozen=True)
class Dirichlet(Differentiable):
    """Dirichlet distribution over the probability simplex $\\Delta^{k-1} = \\{x \\in \\mathbb{R}^k : x_i > 0, \\sum_i x_i = 1\\}$.

    The density is

    $$p(x; \\alpha) = \\frac{1}{B(\\alpha)} \\prod_{i=1}^{k} x_i^{\\alpha_i - 1},$$

    where $B(\\alpha) = \\prod_i \\Gamma(\\alpha_i) / \\Gamma(\\sum_i \\alpha_i)$ is the multivariate beta function.

    As an exponential family:

    - Natural parameter: $\\theta = \\alpha \\in \\mathbb{R}^k_{>0}$ (shape parameters)
    - Sufficient statistic: $s(x) = \\log x$ (component-wise)
    - Base measure: $\\log \\mu(x) = -\\sum_i \\log x_i$
    - Log-partition function: $\\psi(\\alpha) = \\sum_i \\log \\Gamma(\\alpha_i) - \\log \\Gamma(\\sum_i \\alpha_i)$
    - Mean parameter: $\\eta_i = \\mathbb{E}[\\log x_i] = \\psi_{digamma}(\\alpha_i) - \\psi_{digamma}(\\sum_j \\alpha_j)$

    No closed-form ``negative_entropy`` is provided --- recovering $\\alpha$ from $\\mathbb{E}[\\log x]$ requires numerical inversion of the digamma map, so this class is ``Differentiable`` but not ``Analytic``.
    """

    n_categories: int
    """Simplex dimension $k$."""

    @property
    @override
    def dim(self) -> int:
        return self.n_categories

    @property
    @override
    def data_dim(self) -> int:
        return self.n_categories

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return jnp.log(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return -jnp.sum(jnp.log(x))

    @override
    def log_partition_function(self, params: Array) -> Array:
        return jnp.sum(jax.lax.lgamma(params)) - jax.lax.lgamma(jnp.sum(params))

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        return jax.random.dirichlet(key, params, shape=(n,))

    @override
    def initialize(
        self,
        key: Array,
        location: float = 1.0,
        shape: float = 0.1,
    ) -> Array:
        r"""Sample positive shape parameters from $\alpha_i \sim \mathcal{N}(\text{location}, \text{shape}^2)$ clipped at $10^{-3}$."""
        noise = jax.random.normal(key, shape=(self.dim,)) * shape
        return jnp.clip(location + noise, 1e-3, None)

    @override
    def check_natural_parameters(self, params: Array) -> Array:
        """Valid natural parameters are finite and strictly positive."""
        return super().check_natural_parameters(params) & jnp.all(params > 0).astype(
            jnp.int32
        )
