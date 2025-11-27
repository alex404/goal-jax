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
    """Euclidean space $\\mathbb{R}^n$ of dimension $n$. Euclidean space consists of $n$-dimensional real vectors with the standard Euclidean distance metric

    $$d(x,y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}.$$

    Euclidean also serves as the location component of a Normal distribution, and on its own we treat it as a normal distribution with unit covariance.

    As an exponential family:
        - Sufficient statistic: Identity map $s(x) = x$
        - Base measure: $\\mu(x) = -\\frac{n}{2}\\log(2\\pi)$
    """  # Fields

    _dim: int

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Return the dimension of the space."""
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
        """Standard normal base measure including normalizing constant."""
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from a standard normal distribution with mean given by natural parameters.

        For Euclidean space with unit covariance, the natural parameters are the mean,
        so we sample x ~ N(params, I).

        Args:
            key: JAX random key
            params: Natural parameters (the mean)
            n: Number of samples

        Returns:
            Array of shape (n, dim) with samples
        """
        noise = jax.random.normal(key, (n, self.dim))
        return params + noise

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Compute log partition function for standard normal with unit covariance.

        For a normal distribution N(μ, I) with sufficient statistic s(x) = x
        and natural parameter θ = μ:

        ψ(θ) = 0.5 ||θ||² + (d/2) log(2π)

        Args:
            params: Natural parameters (the mean)

        Returns:
            Scalar log partition function value
        """
        return 0.5 * jnp.sum(params**2) + 0.5 * self.dim * jnp.log(2 * jnp.pi)


class GeneralizedGaussian[S: ExponentialFamily](Differentiable, ABC):
    r"""ABC for exponential families with Gaussian-like sufficient statistics.

    This abc captures the shared structure between Normal distributions and
    Boltzmann machines, where the sufficient statistics take the form:

    $$s(x) = (x, x \otimes x)$$

    with appropriate constraints on the second moment term for minimality.

    In theory, both Normal distributions and Boltzmann machines share this fundamental
    sufficient statistic structure but differ in their constraints:
    - Normal: Second moment must be positive definite (continuous domain)
    - Boltzmann: Second moment has off-diagonal interactions only (discrete binary domain)

    This abstraction enables unified conjugation algorithms for harmoniums (bilinear
    exponential family models) that can work with either continuous or discrete
    distributions.

    The location component is always Euclidean space, as both Normal and Boltzmann
    distributions use Euclidean for their first-order sufficient statistics.
    """

    @property
    @abstractmethod
    def loc_man(self) -> Euclidean:
        """Return the Euclidean location component manifold."""

    @property
    @abstractmethod
    def shp_man(self) -> S:
        """Return the shape component manifold."""

    # Core split/join operations for harmonium conjugation

    @abstractmethod
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Split natural parameters into location and precision in natural coordinates.

        For harmonium conjugation, natural coordinates represent:
        - Location: $\\theta_1 = \\Sigma^{-1}\\mu$ (Normal) or bias parameters (Boltzmann)
        - Precision: $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$ (Normal) or $-\\frac{1}{2}J$ (Boltzmann precisions)

        Args:
            params: Parameters in natural coordinates

        Returns:
            location: Location parameters in natural coordinates
            precision: Precision/interaction parameters in natural coordinates
        """

    @abstractmethod
    def join_location_precision(self, location: Array, precision: Array) -> Array:
        """Join location and precision in natural coordinates.

        Inverse of split_location_precision, combining components back into
        full parameter vector.

        Args:
            location: Location parameters in natural coordinates
            precision: Precision/interaction parameters in natural coordinates

        Returns:
            params: Combined parameters in natural coordinates
        """

    @abstractmethod
    def split_mean_second_moment(self, means: Array) -> tuple[Array, Array]:
        """Split parameters into mean and second moment in mean coordinates.

        For harmonium conjugation, mean coordinates represent:
        - Mean: $\\eta_1 = \\mu$ (first moment)
        - Second moment: $\\eta_2 = \\mu\\mu^T + \\Sigma$ (Normal) or correlations (Boltzmann)

        Args:
            params: Parameters in mean coordinates

        Returns:
            mean: Mean parameters in mean coordinates
            second_moment: Second moment parameters in mean coordinates
        """

    @abstractmethod
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Join mean and second moment in mean coordinates.

        Inverse of split_mean_second_moment, combining components back into
        full parameter vector.

        Args:
            mean: Mean parameters in mean coordinates
            second_moment: Second moment parameters in mean coordinates

        Returns:
            params: Combined parameters in mean coordinates
        """
