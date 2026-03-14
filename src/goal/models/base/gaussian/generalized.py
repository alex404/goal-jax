"""Generalized Gaussian abstraction unifying Normal distributions and Boltzmann machines through shared sufficient statistic structure."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import (
    Differentiable,
    ExponentialFamily,
)


@dataclass(frozen=True)
class Euclidean(Differentiable):
    """Euclidean space $\\mathbb{R}^n$ of dimension $n$. Euclidean also serves as the location component of a Normal distribution, and on its own we treat it as a normal distribution with unit covariance.

    As an exponential family:
        - Sufficient statistic: Identity map $s(x) = x$
        - Base measure: $\\mu(x) = -\\frac{n}{2}\\log(2\\pi)$
    """

    # Fields

    _dim: int

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self._dim

    @property
    @override
    def data_dim(self) -> int:
        return self.dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Identity map on the data."""
        return x

    @override
    def log_base_measure(self, x: Array) -> Array:
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from a standard normal with mean given by natural parameters."""
        noise = jax.random.normal(key, (n, self.dim))
        return params + noise

    @override
    def log_partition_function(self, params: Array) -> Array:
        """$\\psi(\\theta) = \\frac{1}{2}\\|\\theta\\|^2 + \\frac{d}{2}\\log(2\\pi)$."""
        return 0.5 * jnp.sum(params**2) + 0.5 * self.dim * jnp.log(2 * jnp.pi)


class GeneralizedGaussian[L: ExponentialFamily, S: ExponentialFamily](
    Differentiable, ABC
):
    """ABC for exponential families with Gaussian-like sufficient statistics $s(x) = (x, x \\\\otimes x)$.

    This captures the shared structure between Normal distributions and Boltzmann machines. Both share this fundamental sufficient statistic structure but differ in their constraints: Normal requires positive definite second moment (continuous domain), while Boltzmann has off-diagonal interactions only (discrete binary domain). This abstraction enables unified conjugation algorithms for harmoniums.
    """

    @property
    @abstractmethod
    def loc_man(self) -> L:
        """Return the location component manifold."""

    @property
    @abstractmethod
    def shp_man(self) -> S:
        """Return the shape component manifold."""

    # Contract

    @abstractmethod
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Split natural parameters into location ($\\\\theta_1 = \\\\Sigma^{-1}\\\\mu$) and precision ($\\\\theta_2 = -\\\\frac{1}{2}\\\\Sigma^{-1}$) components."""

    @abstractmethod
    def join_location_precision(self, location: Array, precision: Array) -> Array:
        """Join location and precision components into natural parameters."""

    @abstractmethod
    def split_mean_second_moment(self, means: Array) -> tuple[Array, Array]:
        """Split mean parameters into first moment ($\\\\eta_1 = \\\\mu$) and second moment ($\\\\eta_2 = \\\\mu\\\\mu^T + \\\\Sigma$) components."""

    @abstractmethod
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Join first and second moment components into mean parameters."""
