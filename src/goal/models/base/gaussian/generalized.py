from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, override

import jax.numpy as jnp
from jax import Array

from ....geometry import ExponentialFamily, Manifold, Mean, Natural, Point


class GeneralizedGaussian[L: Manifold, S: Manifold](ExponentialFamily, ABC):
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

    # Abstract properties

    @property
    @abstractmethod
    def location_manifold(self) -> L:
        """The location component manifold (first-order parameters)."""
        ...

    @property
    @abstractmethod
    def shape_manifold(self) -> S:
        """The shape/interaction component manifold (second-order parameters)."""
        ...

    # Core split/join operations for harmonium conjugation

    @abstractmethod
    def split_location_precision(
        self, params: Point[Natural, Self]
    ) -> tuple[Point[Natural, L], Point[Natural, S]]:
        """Split parameters into location and precision in natural coordinates.

        For harmonium conjugation, natural coordinates represent:
        - Location: θ₁ = Σ⁻¹μ (Normal) or bias parameters (Boltzmann)
        - Precision: θ₂ = -½Σ⁻¹ (Normal) or -½J (Boltzmann interactions)

        Args:
            params: Parameters in natural coordinates

        Returns:
            location: Location parameters in natural coordinates
            precision: Precision/interaction parameters in natural coordinates
        """
        ...

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
        ...

    @abstractmethod
    def split_mean_second_moment(
        self, params: Point[Mean, Self]
    ) -> tuple[Point[Mean, L], Point[Mean, S]]:
        """Split parameters into mean and second moment in mean coordinates.

        For harmonium conjugation, mean coordinates represent:
        - Mean: η₁ = μ (first moment)
        - Second moment: η₂ = μμᵀ + Σ (Normal) or correlations (Boltzmann)

        Args:
            params: Parameters in mean coordinates

        Returns:
            mean: Mean parameters in mean coordinates
            second_moment: Second moment parameters in mean coordinates
        """
        ...

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
        ...

    # Common sufficient statistic implementation

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Compute sufficient statistics s(x) = (x, x⊗x) with appropriate constraints.

        This is the core commonality between Normal and Boltzmann distributions.
        The implementation handles the shared (x, x⊗x) structure while allowing
        subclasses to impose different constraints on the second moment term.

        Args:
            x: Data point

        Returns:
            Sufficient statistics in mean coordinates
        """
        x = jnp.atleast_1d(x)

        # First moment component
        first_moment: Point[Mean, L] = self.location_manifold.point(x)

        # Second moment component - computed by shape manifold
        # This allows Normal vs Boltzmann to handle constraints differently
        second_moment = self._compute_second_moment(x)

        return self.join_mean_second_moment(first_moment, second_moment)

    @abstractmethod
    def _compute_second_moment(self, x: Array) -> Point[Mean, S]:
        """Compute the second moment component with distribution-specific constraints.

        Args:
            x: Data point

        Returns:
            Second moment in mean coordinates with appropriate constraints
        """
