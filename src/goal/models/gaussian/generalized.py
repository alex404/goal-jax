from __future__ import annotations

from typing import Any, Self

from ..geometry import (
    Mean,
    Natural,
    Point,
)


class GeneralizedGaussian[L, S]:
    r"""Protocol for exponential families with Gaussian-like sufficient statistics.

    This protocol captures the shared structure between Normal distributions and
    Boltzmann machines, where the sufficient statistics take the form:

    $$s(x) = (x, x \otimes x)$$

    with appropriate constraints on the second moment term for minimality.
    """

    def split_location_precision(
        self, p: Point[Natural, Any]
    ) -> tuple[Point[Natural, L], Point[Natural, S]]:
        """Split parameters into location and precision in natural coordinates.

        Args:
            p: Parameters in natural coordinates

        Returns:
            loc: Location parameters
            precision: Precision/interaction parameters
        """
        ...

    def join_location_precision(
        self, loc: Point[Natural, Euclidean], precision: Point[Natural, S]
    ) -> Point[Natural, Self]:
        """Join location and precision in natural coordinates.

        Args:
            loc: Location parameters
            precision: Precision/interaction parameters

        Returns:
            p: Combined parameters
        """
        ...

    def split_mean_covariance(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, S]]:
        """Split parameters into mean and covariance in mean coordinates.

        Args:
            p: Parameters in mean coordinates

        Returns:
            mean: Mean parameters
            covariance: Covariance/correlation parameters
        """
        ...

    def join_mean_covariance(
        self, mean: Point[Mean, Euclidean], covariance: Point[Mean, S]
    ) -> Point[Mean, Self]:
        """Join mean and covariance in mean coordinates.

        Args:
            mean: Mean parameters
            covariance: Covariance/correlation parameters

        Returns:
            p: Combined parameters
        """
        ...
