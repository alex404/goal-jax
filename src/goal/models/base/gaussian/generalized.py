from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, override

import jax.numpy as jnp
from jax import Array

from ....geometry import ExponentialFamily, Manifold, Mean, Natural, Point


@dataclass(frozen=True)
class Euclidean(ExponentialFamily):
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
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Identity map on the data."""
        return self.mean_point(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Standard normal base measure including normalizing constant."""
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)


class GeneralizedGaussian[L: ExponentialFamily, S: ExponentialFamily](Manifold, ABC):
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
    """

    # Core split/join operations for harmonium conjugation

    @abstractmethod
    def split_location_precision(
        self, params: Point[Natural, Self]
    ) -> tuple[Point[Natural, L], Point[Natural, S]]:
        """Split parameters into location and precision in natural coordinates.

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
    def join_location_precision(
        self, location: Point[Natural, L], precision: Point[Natural, S]
    ) -> Point[Natural, Self]:
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
    def split_mean_second_moment(
        self, params: Point[Mean, Self]
    ) -> tuple[Point[Mean, L], Point[Mean, S]]:
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
    def join_mean_second_moment(
        self, mean: Point[Mean, L], second_moment: Point[Mean, S]
    ) -> Point[Mean, Self]:
        """Join mean and second moment in mean coordinates.

        Inverse of split_mean_second_moment, combining components back into
        full parameter vector.

        Args:
            mean: Mean parameters in mean coordinates
            second_moment: Second moment parameters in mean coordinates

        Returns:
            params: Combined parameters in mean coordinates
        """
